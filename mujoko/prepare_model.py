from pathlib import Path
import argparse, json, xml.etree.ElementTree as ET
import mujoco, onnxruntime as ort

if __name__ == "__main__":
    p=argparse.ArgumentParser();p.add_argument("urdf",type=Path);p.add_argument("output",type=Path);p.add_argument("--policy",type=Path,required=True);a=p.parse_args()
    metadata=ort.InferenceSession(str(a.policy)).get_modelmeta().custom_metadata_map
    if "fanfan_deployment_config" not in metadata: raise RuntimeError("Policy has no deployment metadata")
    deploy=json.loads(metadata["fanfan_deployment_config"])
    urdf_root=ET.parse(a.urdf).getroot()
    joints=[j.get("name") for j in urdf_root.findall("joint") if j.get("type") not in ("fixed","floating")]
    if set(joints)!=set(deploy["joint_names"]): raise RuntimeError("URDF joints do not match exported policy joints")
    m=mujoco.MjModel.from_xml_path(str(a.urdf.resolve()));tmp=a.output.with_suffix(".converted.xml")
    a.output.parent.mkdir(parents=True,exist_ok=True);mujoco.mj_saveLastXML(str(tmp),m)
    tree=ET.parse(tmp);root=tree.getroot();root.set("model","fanfan_sim2sim")
    option=root.find("option")
    if option is None: option=ET.SubElement(root,"option")
    option.attrib.update(timestep=str(deploy["control"]["sim_dt"]),gravity="0 0 -9.81",integrator="implicitfast")
    world=root.find("worldbody");trunk=world.find("body[@name='Trunk']");trunk.set("pos"," ".join(map(str,deploy["initial_state"]["base_position"])))
    # The copied URDF contains an explicit floating world->Trunk joint. This
    # preserves every original hip origin and inertia without reconstructing
    # the root after MuJoCo has already chosen a reference link.
    for g in trunk.iter("geom"):g.set("contype","1");g.set("conaffinity","0")
    for j in trunk.iter("joint"):
        if j.get("name") != "floating_base":j.set("armature","0.01")
    world.insert(0,ET.Element("geom",{"name":"ground","type":"plane","size":"20 20 0.1","rgba":"0.78 0.80 0.82 1","friction":"1 0.005 0.0001","condim":"3"}))
    world.insert(1,ET.Element("light",{"name":"sun","pos":"0 0 3","dir":"0 0 -1","directional":"true"}))
    act=ET.SubElement(root,"actuator")
    effort={j.get("name"):float(j.find("limit").get("effort")) for j in urdf_root.findall("joint") if j.get("name") in joints}
    for j in joints:
        limit=effort[j]
        ET.SubElement(act,"motor",{"name":j+"_motor","joint":j,"gear":"1","ctrllimited":"true","ctrlrange":f"{-limit:g} {limit:g}"})
    ET.indent(tree,space="  ");tree.write(a.output,encoding="utf-8",xml_declaration=True);tmp.unlink()
    check=mujoco.MjModel.from_xml_path(str(a.output.resolve()));print(f"Prepared nq={check.nq}, nv={check.nv}, nu={check.nu}")
