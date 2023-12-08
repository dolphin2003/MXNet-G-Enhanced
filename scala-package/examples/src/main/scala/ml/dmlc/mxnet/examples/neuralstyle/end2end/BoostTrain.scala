package ml.dmlc.mxnet.examples.neuralstyle.end2end

import org.slf4j.LoggerFactory
import org.kohsuke.args4j.{CmdLineParser, Option}
import scala.collection.JavaConverters._
import ml.dmlc.mxnet.NDArray
import ml.dmlc.mxnet.Shape
import ml.dmlc.mxnet.Context
import ml.dmlc.mxnet.DataBatch
import ml.dmlc.mxnet.Symbol
import ml.dmlc.mxnet.Executor
import ml.dmlc.mxnet.optimizer.SGD
import java.io.File
import javax.imageio.ImageIO
import scala.util.Random
import ml.dmlc.mxnet.optimizer.Adam

/**
 * @author Depeng Liang
 */
object BoostTrain {

  private val logger = LoggerFactory.getLogger(classOf[BoostTrain])

  def getTvGradExecutor(img: NDArray, ctx: Context, tvWeight: Float): Executor = {
    // create TV gradient executor with input binded on img
    if (tvWeight <= 0.0f) null

    val nChannel = img.shape(1)
    val sImg = Symbol.Variable("img")
    val sKernel = Symbol.Variable("kernel")
    val channels = Symbol.SliceChannel()(sImg)(Map("num_outputs" -> nChannel))
    val out = Symbol.Concat()((0 until nChannel).map { i =>
      Symbol.Convolution()()(Map("data" -> channels.get(i), "weight" -> sKernel,
                    "num_filter" -> 1, "kernel" -> "(3,3)", "pad" -> "(1,1)",
                    "no_bias" -> true, "stride" -> "(1,1)"))
    }.toArray: _*)() * tvWeight
    val kernel = {
      val tmp = NDArray.empty(Shape(1, 1, 3, 3), ctx)
      tmp.set(Array[Float](0, -1, 0, -1, 4, -1, 0, -1, 0))
      tmp / 8.0f
    }
    out.bind(ctx, Map("img" -> img, "kernel" -> kernel))
  }

  def main(args: Array[String]): Unit = {
    val stin = new BoostTrain
    val parser: CmdLineParser = new CmdLineParser(stin)
    try {
      parser.parseArgument(args.toList.asJava)
      assert(stin.dataPath != null
          && stin.vggModelPath != null
          && stin.saveModelPath != null
          && stin.styleImage != null)
      // params
      val vggParams = NDArray.load2Map(stin.vggModelPath)
      val styleWeight = 1.2f
      val contentWeight = 10f
      val dShape = Shape(1, 3, 384, 384)
      val clipNorm = 0.05f * dShape.product
      val modelPrefix = "v3"
      val ctx = if (stin.gpu == -1) Context.cpu() else Context.gpu(stin.gpu)

      // init style
      val styleNp = DataProcessing.preprocessStyleImage(stin.styleImage, dShape, ctx)
      var styleMod = Basic.getStyleModule("style", dShape, ctx, vggParams)
      styleMod.forward(Array(styleNp))
      val styleArray = styleMod.getOutputs().map(_.copyTo(Context.cpu()))
      styleMod.dispose()
      styleMod = null

      // content
      val contentMod = Basic.getContentModule("content", dShape, ctx, vggParams)

      // loss
      val (loss, gScale) = Basic.getLossModule("loss", dShape, ctx, vggParams)
      val extraArgs = (0 until styleArray.length)
                                  .map( i => s"target_gram_$i" -> styleArray(i)).toMap
      loss.setParams(extraArgs)
      var gradArray = Array[NDArray]()
      for (i <- 0 until styleArray.length) {
        gradArray = gradArray :+ (NDArray.ones(Shape(1), ctx) * (styleWeight / gScale(i