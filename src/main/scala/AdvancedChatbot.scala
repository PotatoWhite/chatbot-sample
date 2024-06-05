import org.apache.spark.ml.feature._
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, SparkSession}

/**
 * 데이터프레임, 벡터, IDF
 * 데이터프레임 (DataFrame): Spark의 구조화된 데이터 구조로, CSV 파일에서 읽어온 데이터를 저장-> 전처리하여 새로운 컬럼을 추가
 *
 * 벡터 (Vector): 텍스트 데이터를 숫자 형식으로 변환한 것으로, HashingTF와 같은 도구를 사용하여 텍스트를 숫자 벡터로 변환
 *
 * IDF (Inverse Document Frequency): 단어의 중요도를 계산하는 방법으로, 자주 등장하는 단어의 가중치를 낮춥니다.
 * 예: IDF를 사용하여 단어 벡터를 변환하면, 자주 등장하는 단어보다 중요한 단어가 더 큰 가중치를 갖게 됩니다.
 *
 * VectorAssembler: 여러 벡터를 하나로 결합하는 도구  단어와 글자 벡터를 하나의 벡터 텍스트의 결합 -> 특징을 더 잘 파악
 */
object AdvancedChatbot {

  def main(args: Array[String]): Unit = {
    implicit val spark: SparkSession = createSparkSession() // Spark 세션 생성
    val rawDf = loadData(spark) // 데이터 로드 및 전처리
    val data = preprocessData(rawDf) // 데이터 벡터화 및 IDF 변환
    data.combinedData.show() // 전처리된 데이터 출력
    startChatbot(spark, data, rawDf) // 챗봇 시작
    spark.stop() // Spark 세션 종료
  }

  // 형태소 전처리 함수: 한국어의 조사를 제거합니다.
  private def preprocessText(text: String): String = {
    val stopwords = Set("은", "는", "이", "가", "을", "를", "에", "에서", "도", "만", "으로", "로", "와", "과", "하고", "의", "에게", "께서", "뿐")
    text.split(" ").map(word => stopwords.foldLeft(word)((acc, stopword) => acc.replaceAll(stopword + "$", ""))).mkString(" ")
  }

  // 한국어 종결어미를 어간으로 변환하는 함수
  private def convertToEndingsToStems(text: String): String = {
    val endings = Map("습니다" -> "다", "니다" -> "다", "어요" -> "다", "네요" -> "다", "예요" -> "다", "죠" -> "다", "요" -> "다", "네" -> "다", "죠" -> "다", "와요" -> "오다", "워요" -> "우다", "해요" -> "하다", "해" -> "하다", "하자" -> "하다", "하세" -> "하다", "하오" -> "하다", "하니" -> "하다", "하랴" -> "하다", "하거라" -> "하다", "하였다" -> "하다", "하였어" -> "하다", "하였지" -> "하다", "하였으" -> "하다", "하였음" -> "하다", "하였습" -> "하다", "하였소" -> "하다", "하였네" -> "하다", "하였노" -> "하다", "하였다오" -> "하다", "하였다시" -> "하다", "하였다시오" -> "하다")
    text.split(" ").map(word => endings.foldLeft(word)((acc, ending) => acc.replaceAll(ending._1 + "$", ending._2))).mkString(" ")
  }

  // SparkSession 생성
  private def createSparkSession(): SparkSession = {
    val spark = SparkSession.builder
      .appName("AdvancedChatbot")
      .master("local[*]")
      .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")
    spark
  }

  // 데이터 로드 및 전처리
  private def loadData(spark: SparkSession): DataFrame = {
    val dataPath = "src/main/resources/data/chatbot_data.csv"
    val rawDf = spark.read.option("header", "true").csv(dataPath)

    // 한국어 문장 전처리
    rawDf.withColumn("Question", udf((text: String) => convertToEndingsToStems(preprocessText(text))).apply(col("Question")))
  }

  // 전처리된 데이터를 저장하는 케이스 클래스
  case class PreprocessedData(
                               combinedData: DataFrame,
                               wordTokenizer: Tokenizer,
                               wordHashingTF: HashingTF,
                               wordIdfModel: IDFModel,
                               charHashingTF: HashingTF,
                               charIdfModel: IDFModel
                             )

  // 데이터를 전처리하고 벡터화합니다.
  private def preprocessData(df: DataFrame)(implicit spark: SparkSession): PreprocessedData = {
    // 질문을 단어로 나누기 위해 Tokenizer 사용
    val wordTokenizer = new Tokenizer().setInputCol("Question").setOutputCol("words")
    val wordsData = wordTokenizer.transform(df)

    // 질문을 글자 단위로 나누기 위해 UDF 사용
    val charTokenizer = udf((text: String) => text.replaceAll(" ", "").split("").toSeq)
    val charData = df.withColumn("chars", charTokenizer(col("Question")))

    // 단어들을 숫자 벡터로 변환하기 위해 HashingTF 사용
    val wordHashingTF = new HashingTF().setInputCol("words").setOutputCol("wordFeatures").setNumFeatures(1000)
    val wordFeaturizedData = wordHashingTF.transform(wordsData)

    // 글자들을 숫자 벡터로 변환하기 위해 HashingTF 사용
    val charHashingTF = new HashingTF().setInputCol("chars").setOutputCol("charFeatures").setNumFeatures(1000)
    val charFeaturizedData = charHashingTF.transform(charData)

    // 단어 벡터를 IDF로 변환
    val wordIDF = new IDF().setInputCol("wordFeatures").setOutputCol("wordRescaledFeatures")
    val wordIdfModel = wordIDF.fit(wordFeaturizedData)
    val wordRescaledData = wordIdfModel.transform(wordFeaturizedData)

    // 글자 벡터를 IDF로 변환
    val charIDF = new IDF().setInputCol("charFeatures").setOutputCol("charRescaledFeatures")
    val charIdfModel = charIDF.fit(charFeaturizedData)
    val charRescaledData = charIdfModel.transform(charFeaturizedData)

    // 단어 벡터와 글자 벡터를 결합
    val assembler = new VectorAssembler()
      .setInputCols(Array("wordRescaledFeatures", "charRescaledFeatures"))
      .setOutputCol("features")
    val combinedData = assembler.transform(wordRescaledData.join(charRescaledData, "Question"))

    PreprocessedData(combinedData, wordTokenizer, wordHashingTF, wordIdfModel, charHashingTF, charIdfModel)
  }

  // 챗봇을 시작합니다.
  def startChatbot(spark: SparkSession, data: PreprocessedData, df: DataFrame): Unit = {
    import spark.implicits._
    var continueChatting = true

    while (continueChatting) {
      println("질문을 해주세요 (종료하려면 '종료' 입력):")
      val userInput = scala.io.StdIn.readLine().toLowerCase()

      if (userInput == "종료") {
        continueChatting = false
      } else {
        // 사용자 입력 전처리
        val preprocessedUserInput = convertToEndingsToStems(preprocessText(userInput))

        // 사용자 입력을 데이터프레임으로 변환
        val userDF = Seq((preprocessedUserInput, "")).toDF("Question", "Response")

        // 사용자 입력을 단어로 나누어 벡터로 변환
        val userWordsData = data.wordTokenizer.transform(userDF)

        // 사용자 입력을 글자로 나누어 벡터로 변환
        val charTokenizer = udf((text: String) => text.replaceAll(" ", "").split("").toSeq) // 글자로 나누는 함수
        val userCharData = userDF.withColumn("chars", charTokenizer(col("Question"))) // 글자로 나눈 데이터프레임

        // 사용자 입력을 벡터로 변환
        val userWordFeaturizedData = data.wordHashingTF.transform(userWordsData)
        val userCharFeaturizedData = data.charHashingTF.transform(userCharData)

        // 사용자 입력을 IDF로 변환
        val userWordRescaledData = data.wordIdfModel.transform(userWordFeaturizedData)
        val userCharRescaledData = data.charIdfModel.transform(userCharFeaturizedData)

        // 사용자 입력을 단어 벡터와 글자 벡터로 결합
        val assembler = new VectorAssembler()
          .setInputCols(Array("wordRescaledFeatures", "charRescaledFeatures"))
          .setOutputCol("features")
        val userCombinedData = assembler.transform(userWordRescaledData.join(userCharRescaledData, "Question"))

        // 사용자 입력의 특징 벡터 추출
        val userFeatures = userCombinedData.select("features").head().getAs[Vector]("features")

        // 코사인 유사도를 계산 함수
        val cosSim = (v1: Vector, v2: Vector) => {
          val dotProduct = v1.toArray.zip(v2.toArray).map { case (x, y) => x * y }.sum
          val norm1 = math.sqrt(v1.toArray.map(math.pow(_, 2)).sum)
          val norm2 = math.sqrt(v2.toArray.map(math.pow(_, 2)).sum)
          dotProduct / (norm1 * norm2)
        }

        // 유사도 계산
        val similarities = data.combinedData.select("Question", "features").as[(String, Vector)].map {
          case (question, features) => (question, cosSim(features, userFeatures)) // 사용자 입력과의 유사도 계산
        }.toDF("Question", "Similarity")

        // 유사도가 가장 높은 질문 선택
        val mostSimilar = similarities.orderBy(desc("Similarity")).first()

        // 선택된 질문과 유사한 질문에 대한 응답 출력
        val response = df.filter($"Question" === mostSimilar.getString(0)).select("Response").head().getString(0)
        println(response)
        println(s"(선택된 이유: '${mostSimilar.getString(0)}' 질문과의 유사도 ${mostSimilar.getDouble(1)})")
      }
    }
  }
}

