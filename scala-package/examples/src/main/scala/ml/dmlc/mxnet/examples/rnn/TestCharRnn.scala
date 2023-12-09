package ml.dmlc.mxnet.examples.rnn

import ml.dmlc.mxnet._
import org.kohsuke.args4j.{CmdLineParser, Option}
import org.slf4j.LoggerFactory
import scala.collection.JavaConverters._

/**
 * Follows the demo, to test the char rnn:
 * https://github.com/dmlc/mxnet/blob/master/example/rnn/char-rnn.ipynb
 * @author Depeng Liang
 */
object TestCharRnn {

  private val logger = LoggerFactory.getLogger(classOf[TrainCharRnn])

  def main(args: Array[String]): Unit = {
    val stcr = new TestCharRnn
    val parser: CmdLineParser = new CmdLineParser(stcr)
    try {
      parser.parseArgument(args.toList.asJava)
      assert(stcr.dataPath != null && stcr.modelPrefix != null && stcr.starterSentence != null)

      // The batch size for training
      val batchSize = 32
      // We can support various length input
      // For this problem, we cut each input sentence to length of 129
      // So we only need fix length bucket
      val buckets = List(129)
      // hidden unit in LSTM cell
      val numHidden = 512
      // embedding dimension, which is, map a char to a 256 dim vector
      val numEmbed = 256
      // number of lstm layer
      val numLstmLayer = 3

      // build char vocabluary from input
      val vocab = Utils.buildVocab(stcr.dataPath)

      // load from check-point
      val (_, argParams, _) = Model.loadCheckpoint(stcr.modelPrefix, 75)

      // build an inference model
      val model = new RnnModel.LSTMInferenceModel(numLstmLayer, vocab.size + 1,
                           numHidden = numHidden, numEmbed = numEmbed,
                           numLabel = vocab.size + 1, argParams = argParams, dropout = 0.2f)

      // generate a sequence of 1200 chars
      val seqLength = 1200
      val inputNdarray = NDArray.zeros(1)
      val revertVocab = Utils.makeRevertVoca