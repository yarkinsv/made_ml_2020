name := "lsml_hw7"

version := "0.1"

scalaVersion := "2.12.12"

val sparkVersion = "3.0.1"
val scalacticVersion = "3.2.2"

libraryDependencies += "org.apache.spark" %% "spark-sql" % sparkVersion
libraryDependencies += "org.apache.spark" %% "spark-mllib" % sparkVersion
libraryDependencies += "org.scalactic" %% "scalactic" % scalacticVersion
libraryDependencies += "org.scalatest" %% "scalatest" % scalacticVersion % "test"
