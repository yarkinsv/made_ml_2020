package org.apache.spark.ml.made

import org.apache.spark.ml.attribute.AttributeGroup
import org.apache.spark.ml.linalg.{DenseVector, Vector, Vectors}
import org.apache.spark.ml.param.shared.HasMaxIter
import org.apache.spark.ml.param.{DoubleParam, ParamMap, ParamValidators}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.ml.{Estimator, Model, PredictorParams}
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset, Encoder}


trait LinearRegressionParams extends PredictorParams with HasMaxIter {
  def setFeaturesCol(value: String) : this.type = set(featuresCol, value)
  def setPredictionCol(value: String): this.type = set(predictionCol, value)
  def setLabelCol(value: String): this.type = set(labelCol, value)

  def setMaxIter(value: Int): this.type = set(maxIter, value)

  final val stepSize: DoubleParam = new DoubleParam(this, "stepSize", "Step size " +
    "(a.k.a. learning rate) in interval (0, 1] for shrinking the contribution of each estimator.",
    ParamValidators.inRange(0, 1, lowerInclusive = false, upperInclusive = true))

  def setStepSize(value: Double): this.type = set(stepSize, value)
}

class LinearRegression(override val uid: String) extends Estimator[LinearModel] with LinearRegressionParams {

  def this() = this(Identifiable.randomUID("linearRegression"))

  override def fit(dataset: Dataset[_]): LinearModel = {
    implicit val encoder : Encoder[Tuple2[Vector, Double]] = ExpressionEncoder()

    val vectors = dataset.select($(featuresCol), $(labelCol)).as[Tuple2[Vector, Double]]

    val dim: Int = AttributeGroup.fromStructField(dataset.schema($(featuresCol))).numAttributes.getOrElse(
      vectors.first()._1.size
    )

    var w = Vectors.zeros(dim).asBreeze
    var b = 0.0

    for (_ <- 0 to $(maxIter)) {
      val deltaL = vectors.rdd
        .map(row => {
          val x = row._1.asBreeze
          val y = row._2

          val delta_b = $(stepSize) * -2.0 * (y - ((w dot x) + b))
          val delta_w = delta_b * x

          (delta_w, delta_b)
        })
        .reduce((row1, row2) => {
          (row1._1 + row2._1, row1._2 + row2._2)
        })

      w = w - deltaL._1
      b = b - deltaL._2
    }

    val model = new LinearModel(
      coef = new DenseVector(w.toArray),
      intercept = b
    )
    model
  }

  override def copy(extra: ParamMap): Estimator[LinearModel] = ???

  override def transformSchema(schema: StructType): StructType = ???

}

class LinearModel private[made](override val uid: String,
                                val coef: DenseVector,
                                val intercept: Double) extends Model[LinearModel] with LinearRegressionParams {

  private[made] def this(coef: DenseVector, intercept: Double) =
    this(Identifiable.randomUID("linearModel"), coef, intercept)

  override def copy(extra: ParamMap): LinearModel = ???

  override def transform(dataset: Dataset[_]): DataFrame = {
    val transformUdf = dataset.sqlContext.udf.register(uid + "_transform",
      (x : Vector) => {
        Vectors.dense((x.asBreeze dot coef.asBreeze) + intercept)
      })

    dataset.withColumn($(predictionCol), transformUdf(dataset($(featuresCol))))
  }

  override def transformSchema(schema: StructType): StructType = ???
}
