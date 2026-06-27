from pathlib import Path
import argparse, xml.etree.ElementTree as ET
import mujoco

JOINTS=[f"{leg}_{joint}_joint" for leg in ("FL","FR","RL","RR") for joint in ("hip","thigh","calf")]
if __name__ == "__main__":
    p=argparse.ArgumentParser();p.add_argument("urdf",type=Path);p.add_argument("output",type=Path);a=p.parse_args()
    m=mujoco.MjModel.from_xml_path(str(a.urdf.resolve()));tmp=a.output.with_suffix(".converted.xml")
    a.output.parent.mkdir(parents=True,exist_ok=True);mujoco.mj_saveLastXML(str(tmp),m)
    tree=ET.parse(tmp);root=tree.getroot();root.set("model","fanfan_sim2sim")
    option=root.find("option") or ET.SubElement(root,"option");option.attrib.update(timestep="0.005",gravity="0 0 -9.81",integrator="implicitfast")
    world=root.find("worldbody");trunk=world.find("body[@name='Trunk']");trunk.set("pos","0 0 0.295")
    # The copied URDF contains an explicit floating world->Trunk joint. This
    # preserves every original hip origin and inertia without reconstructing
    # the root after MuJoCo has already chosen a reference link.
    for g in trunk.iter("geom"):g.set("contype","1");g.set("conaffinity","0")
    for j in trunk.iter("joint"):
        if j.get("name") != "floating_base":j.set("armature","0.01")
    world.insert(0,ET.Element("geom",{"name":"ground","type":"plane","size":"20 20 0.1","rgba":"0.78 0.80 0.82 1","friction":"1 0.005 0.0001","condim":"3"}))
    world.insert(1,ET.Element("light",{"name":"sun","pos":"0 0 3","dir":"0 0 -1","directional":"true"}))
    act=ET.SubElement(root,"actuator")
    for j in JOINTS:ET.SubElement(act,"motor",{"name":j+"_motor","joint":j,"gear":"1","ctrllimited":"true","ctrlrange":"-17 17"})
    ET.indent(tree,space="  ");tree.write(a.output,encoding="utf-8",xml_declaration=True);tmp.unlink()
    check=mujoco.MjModel.from_xml_path(str(a.output.resolve()));print(f"Prepared nq={check.nq}, nv={check.nv}, nu={check.nu}")
