from typing import Dict, List, Optional
import logging

from overrides import overrides
import tqdm

from allennlp.common import Params
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.fields import TextField, LabelField, SequenceLabelField
from allennlp.data.tokenizers.token import Token
from allennlp.data.fields.field import Field  # pylint: disable=unused-import
# from allennlp.modules.token_embedders import PretrainedTransformerEmbedder
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.data.token_indexers import PretrainedTransformerIndexer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@DatasetReader.register("BertDatasetReader")
class BertDatasetReader(DatasetReader):
    """
    Reads a file from ProPara state change dataset.  This data is formatted as TSV, one instance per line.
    Format: "Query \t\t\t step \t\t\t state_change_types"
    state_change_types: string label applicable to this datapoint

    We convert these columns into fields named 
    "tokens", 
    "state_change_types".

    Parameters
    ----------
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": BertTokenIndexer()}``)
    IMPORTANT NOTE: All components like tokeniser, token-indexer, token embedder and Seq2VecEncoder should be of Bert-type.
    """

    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self.transformer_model = "bert-base-uncased"
        self.tokenizer = PretrainedTransformerTokenizer(model_name=self.transformer_model,add_special_tokens=False,max_length=512)
        self.token_indexer = PretrainedTransformerIndexer(model_name=self.transformer_model,max_length =512)
#         self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        instances = []
        with open(file_path, 'r') as state_change_file:
            logger.info("Reading state change instances from TSV dataset at: %s", file_path)
            for line in tqdm.tqdm(state_change_file):
                parts: List[str] = line.split('\t\t\t')
                query_text = parts[0].lower()
                query_tokens = self.tokenizer.tokenize(query_text)
                step_text = parts[1].lower()
                step_tokens = self.tokenizer.tokenize(step_text)
                combined_tokens = self.tokenizer.add_special_tokens(query_tokens,step_tokens)

                # parse labels
                state_change_types = parts[2].strip()

                # create instance
                yield self.text_to_instance(combined_tokens=combined_tokens,
                                                       state_change_types=state_change_types)


    @overrides
    def text_to_instance(self,  # type: ignore
                         combined_tokens: List[str],
                         state_change_types: Optional[List[str]] = None) -> Instance:
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}
#         print(verb_vector)
        # encode inputs
        token_field = TextField(combined_tokens, {'tokens': self.token_indexer})
#         token_field.index(vocab)
        fields['tokens'] = token_field
#         fields['verb_span'] = SequenceLabelField(verb_vector, token_field, 'indicator_tags')
#         fields['entity_span'] = SequenceLabelField(entity_vector, token_field, 'indicator_tags')

        # encode outputs
        if state_change_types:
            fields['state_change_type_labels'] = LabelField(state_change_types, 'state_change_type_labels')
        return Instance(fields)

    @classmethod
    def from_params(cls, params: Params,constructor_to_call=None, constructor_to_inspect=None) -> 'BertDatasetReader':
#         token_indexers = TokenIndexer()
#         print(params.pop("token_indexer", {}))
#         token_indexers = {'tokens': SingleIdTokenIndexer()}
#         params.assert_empty(cls.__name__)
        return BertDatasetReader()
