from typing import Dict, List

import torch
import torch.nn.functional as F
from torch.nn import Linear
from overrides import overrides
import numpy

from allennlp.common import Params
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder, Attention, TimeDistributed, Seq2VecEncoder
from allennlp.nn import InitializerApplicator
from allennlp.nn.util import get_text_field_mask, weighted_sum
from allennlp.training.metrics import F1Measure, CategoricalAccuracy
# from allennlp.modules.similarity_functions.bilinear import BilinearSimilarity
from allennlp.training.metrics import SpanBasedF1Measure
from allennlp.nn.util import sequence_cross_entropy_with_logits
from allennlp.modules.token_embedders import Embedding, PretrainedTransformerEmbedder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.attention import BilinearAttention

@Model.register("BertModel")
class BertModel(Model):
    """
    IMPORTANT NOTE: All components like tokeniser, token-indexer, token embedder and Seq2VecEncoder should be of Bert-type.
    This ``Model`` takes as input a dataset read by stateChangeDatasetReader
    Input: Natural Text Query appended with the step text
    Output: state change types for the focus entity
    The basic outline of this model is to 
        1. Pass the tokenised text through Bert (using Seq2VecEncoder)
        2. Train a linear classifier over CLS embedding
  
    Parameters
    ----------
    vocab : ``Vocabulary``
    text_field_embedder : ``TextFieldEmbedder``
        Used to embed the ``sentence_tokens`` ``TextFields`` we get as input to the model.
    seq2vec_encoder : ``Seq2VecEncoder``
        The encoder to pass text through Bert (or Bert-like) model, and return CLS embedding
    initializer : ``InitializerApplicator``
        We will use this to initialize the parameters in the model, calling ``initializer(self)``.

    Sample commandline
    ------------------
    python propara/run.py train -s /output_folder experiment_config/state_change_local.json 
    """

    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 seq2vec_encoder: Seq2VecEncoder,
                 initializer: InitializerApplicator) -> None:
        super(BertModel, self).__init__(vocab)

        self.text_field_embedder = text_field_embedder
        self.seq2vec_encoder = seq2vec_encoder

        self.num_types = self.vocab.get_vocab_size("state_change_type_labels")
        self.aggregate_feedforward = Linear(seq2vec_encoder.get_output_dim(),
                                            self.num_types)

        self._type_accuracy = CategoricalAccuracy()

        self.type_f1_metrics = {}
        self.type_labels_vocab = self.vocab.get_index_to_token_vocabulary("state_change_type_labels")
        for type_label in self.type_labels_vocab.values():
            self.type_f1_metrics["type_" + type_label] = F1Measure(self.vocab.get_token_index(type_label, "state_change_type_labels"))

        self._loss = torch.nn.CrossEntropyLoss()
        initializer(self)

    def forward(self,  # type: ignore
                tokens: Dict[str, torch.LongTensor],
                state_change_type_labels: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        tokens : Dict[str, torch.LongTensor], required
            The output of ``TextField.as_array()``, which should typically be passed directly to a
            ``TextFieldEmbedder``. This output is a dictionary mapping keys to ``TokenIndexer``
            tensors.  At its most basic, using a ``SingleIdTokenIndexer`` this is: ``{"tokens":
            Tensor(batch_size, num_tokens)}``. This dictionary will have the same keys as were used
            for the ``TokenIndexers`` when you created the ``TextField`` representing your
            sequence.  The dictionary is designed to be passed directly to a ``TextFieldEmbedder``,
            which knows how to combine different word representations into a single vector per
            token in your input. 
        state_change_type_labels: torch.LongTensor, optional (default = None)
            A torch tensor representing the state change type class labels of shape ''(batch_size, 1)''

        Returns
        -------
        An output dictionary consisting of:
        type_probs : torch.FloatTensor
            A tensor of shape ``(batch_size, num_state_change_types)`` representing
            a distribution of state change types per datapoint.
        tags_class_probabilities : torch.FloatTensor
            A tensor of shape ``(batch_size, num_state_change_types, num_tokens)`` representing
            a distribution of location tags per token in a sentence.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        """

        # Layer 1 = Bert embedding layer
        embedded_sentence = self.text_field_embedder(tokens)
        mask = get_text_field_mask(tokens).float()
        
        # Layer 2 = Contextual embedding layer using Bi-LSTM over the sentence
        contextual_embedding = self.seq2vec_encoder(embedded_sentence, mask)

        # Layer 3 = Dense softmax layer to pick one state change type per datapoint,
        # and one tag per word in the sentence
        type_logits = self.aggregate_feedforward(contextual_embedding)
        type_probs = torch.nn.functional.softmax(type_logits, dim=-1)

        # Create output dictionary for the trainer
        # Compute loss and epoch metrics
        output_dict = {'type_probs': type_probs}
        if state_change_type_labels is not None:
            state_change_type_labels_loss = self._loss(type_logits, state_change_type_labels.long().view(-1))
            for type_label in self.type_labels_vocab.values():
                metric = self.type_f1_metrics["type_" + type_label]
                metric(type_probs, state_change_type_labels)

            self._type_accuracy(type_probs, state_change_type_labels)

        output_dict['loss'] = (state_change_type_labels_loss)

        return output_dict

#     @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        predict most probable type labels
        """
        type_predictions = output_dict['type_probs']
        # batch_size, #classes=4
        type_predictions = type_predictions.cpu().data.numpy()
        argmax_indices = numpy.argmax(type_predictions, axis=-1)
        type_labels = [self.vocab.get_token_from_index(x, namespace="state_change_type_labels")
                       for x in argmax_indices]
        output_dict['predicted_types'] = type_labels
        return output_dict

    def get_metrics(self, reset: bool = False):
        metric_dict = {}#self.span_metric.get_metric(reset=reset)

        type_accuracy = self._type_accuracy.get_metric(reset)
        metric_dict['type_accuracy'] = type_accuracy

        for name, metric in self.type_f1_metrics.items():
            metric_val = metric.get_metric(reset)
            metric_dict[name + '_P'] = metric_val[0]
            metric_dict[name + '_R'] = metric_val[1]
            metric_dict[name + '_F1'] = metric_val[2]

        metric_dict['combined_metric'] = type_accuracy
#         print(metric_dict)
#         exit(0)
        return metric_dict

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params,constructor_to_call=None, constructor_to_inspect=None) -> 'BertModel':
        #initialize the class using JSON params
        embedder_params = params.pop("text_field_embedder")
        token_params = embedder_params.pop("tokens")
        embedding = PretrainedTransformerEmbedder.from_params(vocab=vocab,params=token_params)
        text_field_embedder = BasicTextFieldEmbedder(token_embedders={'tokens': embedding})
#         text_field_embedder = TextFieldEmbedder.from_params(vocab, embedder_params)
        
        seq2vec_encoder_params = params.pop("seq2vec_encoder")
        seq2vec_encoder = Seq2VecEncoder.from_params(seq2vec_encoder_params)

        initializer = InitializerApplicator()#.from_params(params.pop("initializer", []))

        params.assert_empty(cls.__name__)
#         print(cls)
        return cls(vocab=vocab,
                   text_field_embedder=text_field_embedder,
                   seq2vec_encoder=seq2vec_encoder,
                   initializer=initializer)
