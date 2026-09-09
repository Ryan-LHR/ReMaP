# -*-coding:utf-8-*-

from baselines.coverage.simple_coverage import *
from baselines.simple_others import *
from baselines.surprise.surprise import *
from baselines.confidence.lof.lof import prioritize_by_lof
from baselines.confidence.simple_confidence import prioritize_by_confidence
from baselines.confidence.mcp_official.mcp_official import prioritize_by_mcp_official
from baselines.confidence.datis.datis import prioritize_by_datis
from baselines.confidence.ats.ats import prioritize_by_ats
from baselines.confidence.nns.nns import prioritize_by_nns
from baselines.confidence.nss.nss import prioritize_by_nss
from baselines.confidence.certpri.certpri import prioritize_by_certpri
from baselines.confidence.rts.rts import prioritize_by_rts
from baselines.confidence.fast.fast import prioritize_by_fast
from baselines.confidence.prima.prima import prioritize_by_prima
from baselines.confidence.sets.sets import prioritize_by_sets
