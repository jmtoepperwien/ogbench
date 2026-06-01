from ogbench.impls.agents.crl import CRLAgent
from ogbench.impls.agents.gcbc import GCBCAgent
from ogbench.impls.agents.gciql import GCIQLAgent
from ogbench.impls.agents.gcivl import GCIVLAgent
from ogbench.impls.agents.hiql import HIQLAgent
from ogbench.impls.agents.mqe import MQEAgent
from ogbench.impls.agents.qrl import QRLAgent
from ogbench.impls.agents.sac import SACAgent
from ogbench.impls.agents.cmd import CMDAgent

agents = dict(
    crl=CRLAgent,
    gcbc=GCBCAgent,
    gciql=GCIQLAgent,
    gcivl=GCIVLAgent,
    hiql=HIQLAgent,
    mqe=MQEAgent,
    qrl=QRLAgent,
    sac=SACAgent,
    cmd=CMDAgent,
)
