ThisBuild / version := "0.1.0-SNAPSHOT"

ThisBuild / scalaVersion := "2.12.19"

lazy val root = (project in file("."))
  .settings(
    name := "chatbot-sample",
    libraryDependencies ++= Seq(
      "org.apache.spark" %% "spark-core" % "3.1.2",
      "org.apache.spark" %% "spark-sql" % "3.1.2",
      "org.apache.spark" %% "spark-mllib" % "3.1.2",
      "org.bitbucket.eunjeon" %% "seunjeon" % "1.5.0"
    )
)


