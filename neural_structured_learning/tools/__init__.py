"""Tools and APIs for preparing data for Neural Structured Learning.

In addition to the functions exported here, two of the modules can be invoked
from the command-line.

Sample usage for running the graph builder:

`python -m neural_structured_learning.tools.build_graph` [*flags*]
*embedding_file.tfr... output_graph.tsv*

Sample usage for preparing input for graph-based NSL:

`python -m neural_structured_learning.tools.pack_nbrs` [*flags*]
*labeled.tfr unlabeled.tfr graph.tsv output.tfr*

For details about these programs' flags, run these commands:

```sh
$ python -m neural_structured_learning.tools.build_graph --help
$ python -m neural_structured_learning.tools.pack_nbrs --help
```
"""

from neural_structured_learning.tools.build_graph import build_graph
from neural_structured_learning.tools.build_graph import build_graph_from_config
from neural_structured_learning.tools.graph_utils import add_edge
from neural_structured_learning.tools.graph_utils import add_undirected_edges
from neural_structured_learning.tools.graph_utils import read_tsv_graph
from neural_structured_learning.tools.graph_utils import write_tsv_graph
from neural_structured_learning.tools.pack_nbrs import pack_nbrs
