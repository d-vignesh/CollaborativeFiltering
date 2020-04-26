import org.apache.spark._
import org.apache.spark.SparkContext._
import scala.io.Codec
import java.nio.charset.CodingErrorAction
import scala.io.Source
import scala.math.sqrt

object CF_Implementation {

	type MovieRating = (Int, Double)
	type UserRatingPair = (Int, (MovieRating, MovieRating))

	type RatingPair = (Double, Double)
	type RatingPairs = Iterable[RatingPair]

	type UserRate = (Int, MovieRating)

	type MappedMovieRating = ((Int, Int), RatingPair)

	def getMovieNames(): Map[Int, String] = {

		// what does this line do
		implicit val codec = Codec("UTF-8")
		codec.onMalformedInput(CodingErrorAction.REPLACE)
		codec.onUnmappableCharacter(CodingErrorAction.REPLACE)

		var movName: Map[Int, String] = Map()
		val lines = Source.fromFile("/home/vignesh/my_drive/Data Science/spark/CollaborativeFilteringImplementationInScala/ml-25m/movies.csv").getLines().drop(1)

		// println("lines dtype : ")
		// println(lines.getClass)

		for(line <- lines) {
			val rec = line.split(",")
			if (rec.length > 0) {
				movName += (rec(0).toInt -> rec(1))
			}
		}

		return movName
	}

	def userBasedMapping(line: String): UserRate = {
		val features = line.split(",")
		val usr = features(0).toInt
		val movie = features(1).toInt
		val rating = features(2).toDouble

		return (usr, (movie, rating)) 
	}

	def filterDuplicate(usrRatings: UserRatingPair): Boolean = {
		val mov1 = usrRatings._2._1._1
		val mov2 = usrRatings._2._2._1
		return mov1 != mov2
	}

	def mappPair(usrRatings: UserRatingPair): MappedMovieRating = {
		val movRate1 = usrRatings._2._1
		val movRate2 = usrRatings._2._2

		val mov1 = movRate1._1
		val rate1 = movRate1._2
		val mov2 = movRate2._1
		val rate2 = movRate2._2

		return ((mov1, mov2), (rate1, rate2))
	}

	def computeCosineSimilarity(ratingPairs: RatingPairs): (Double, Int) = {
		var numPairs: Int = 0
		var sum_xx: Double = 0.0
		var sum_yy: Double = 0.0
		var sum_xy: Double = 0.0

		for (pair <- ratingPairs) {
			val ratingX = pair._1
			val ratingY = pair._2

			sum_xx += ratingX * ratingX
			sum_yy += ratingY * ratingY
			sum_xy += ratingX * ratingY
			numPairs += 1
		}

		val numerator: Double = sum_xy
		val denominator = sqrt(sum_xx) * sqrt(sum_yy)

		var score: Double = 0.0
		if (denominator != 0) {
			score = numerator / denominator
		}

		return(score, numPairs)
	}

	def main(args: Array[String]) {

		val sc = new SparkContext("local[*]", "CollaborativeFiltering")
		sc.setLogLevel("ERROR")

		println("loading movie names ... ")
		val namesDict = getMovieNames()

		println("loading ratings data ... ")
		val data = sc.textFile("/home/vignesh/my_drive/Data Science/spark/CollaborativeFilteringImplementationInScala/ml-25m/ratings.csv")
		// println(data.getClass) // user_id, movie_id, rating, timestamp
		println("total number of ratings : ", data.count())

		val header = data.first()
		val ratings = data.filter(line => !(line.contains(header)))

		val usrRating = ratings.map(userBasedMapping) // (usr, (mov, rate))
		// usrRating.take(10).foreach(println)
		println("usrRating count : ", usrRating.count())

		val joinedRating = usrRating.join(usrRating) // (usr, ((mov, rate), (mov, rate)))
		// println(joinedRating.getClass)
		println("joinedRating count : ", joinedRating.count())

		val uniqueRating = joinedRating.filter(filterDuplicate)
		println("uniqueRating count : ", uniqueRating.count())

		val mappedPairs = uniqueRating.map(mappPair) // ((mov1, mov2), (rate1, rate2))

		val groupedByMovie = mappedPairs.groupByKey()

		val movieSimilarity = groupedByMovie.mapValues(computeCosineSimilarity).cache()
		// println("movieSimilarity rdd : ")
		// println(movieSimilarity.take(50).foreach(println))

		val scoreThreshold: Double = 0.97
		val coOccurenceThreshold: Double = 50.0

		if (args.length > 0) {
			val movieID: Int = args(0).toInt

			val filteredResults = movieSimilarity.filter( x =>
				{
					val pair = x._1
					val sim = x._2
					(pair._1 == movieID || pair._2 == movieID) && sim._1 > scoreThreshold && sim._2 > coOccurenceThreshold
				}
			)

			val results = filteredResults.map(x => (x._2, x._1)).sortByKey(false).take(10)

			println("\nTop 10 similar movies for " + namesDict(movieID))
			for(result <- results) {
				val sim = result._1
				val pair = result._2

				var similarMovieID = pair._1
				if( similarMovieID == movieID) {
					similarMovieID = pair._2
				}
				println(namesDict(similarMovieID) + "\tscore: " + sim._1 + "\tstrength: " + sim._2)
			}
		}
	}
}