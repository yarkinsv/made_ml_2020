package org.apache.spark.ml.made

import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should


class LinearRegressionTest extends AnyFlatSpec with should.Matchers {

  val delta = 0.001

  "Model" should "make predictions" in {
    val model = new LinearModel(
      coef = Vectors.dense(2.0, -0.5).toDense,
      intercept = 0.5
    )
      .setFeaturesCol("features")
      .setPredictionCol("prediction")

    val spark = SparkSession.builder
      .appName("Simple Application")
      .master("local[4]")
      .getOrCreate()

    val sqlc = spark.sqlContext

    import sqlc.implicits._

    val data: DataFrame = Seq(
      Tuple1(Vectors.dense(13.5, 12)),
      Tuple1(Vectors.dense(-1, 0))
    ).toDF("features")

    val vectors: Array[Vector] = model.transform(data).collect().map(_.getAs[Vector]("prediction"))

    vectors.length should be(2)

    vectors(0)(0) should be((13.5 * 2.0) + (12 * -0.5) + 0.5 +- delta)
    vectors(1)(0) should be((-1 * 2.0) + (0 * -0.5) + 0.5 +- delta)
  }

  "Estimator" should "fit the model" in {
    val estimator = new LinearRegression()
      .setFeaturesCol("features")
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMaxIter(100)
      .setStepSize(0.02)

    val spark = SparkSession.builder
      .appName("Simple Application")
      .master("local[4]")
      .getOrCreate()

    val sqlc = spark.sqlContext

    import sqlc.implicits._

    val data: DataFrame = Seq(
      Tuple2(Vectors.dense(0.0), 1.0),
      Tuple2(Vectors.dense(1.0), 3.0),
      Tuple2(Vectors.dense(2.0), 5.0),
      Tuple2(Vectors.dense(3.0), 7.0)
    ).toDF("features", "label")

    val model = estimator.fit(data)

    model.coef(0) should be(2.0 +- delta)
    model.intercept should be(1.0 +- delta)
  }
}
