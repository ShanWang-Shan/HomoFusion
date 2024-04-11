from .common import BaseViz
from ..data.apolloscape.trainId2color import labels


class ApolloscapeViz(BaseViz):
    class_names = [l.name for l in labels]
    SEMANTICS = class_names
