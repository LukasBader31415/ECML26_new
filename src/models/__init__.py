from .M3GLVQ_global_old import M3GLVQ as GlobalOld
from .M3GLVQ_global import M3GLVQ_Global as GlobalNew
from .M3GLVQ_label_old import M3GLVQ_Label as LabelOld
from .M3GLVQ_label import M3GLVQ_Label as LabelNew

M3GLVQ_Global = GlobalNew
M3GLVQ_Label = LabelNew

from .proto_init import init_class_prototypes, DegenerateInitError
from .normalize import build_normalized_view_list, normalize_views, build_m3glvq_view_dict