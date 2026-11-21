"""Write explicit simulator approvals and a Gym-qualified subset after auditing."""
import copy
import hashlib
import json
import os
import shutil
from pathlib import Path
from complete_v3 import BASE,ROOT


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    root=BASE/'amp_reference_v3';audit=json.loads((root/'dataset_audit.json').read_text());assert audit['passed']
    fk=json.loads((BASE/'gym_fk_all_separated/result.json').read_text());assert fk['passed']
    fk_log=(BASE/'gym_fk_all_separated.log').read_text();assert 'invalid parameter' not in fk_log
    transitions=json.loads((BASE/'gym_transitions/transition_result.json').read_text());tgood={r['clip']:r for r in transitions['cases'] if r['passed']}
    m=json.loads((root/'manifest.json').read_text());tm=json.loads((root/'transitions/manifest.json').read_text())
    m.update(kinematic_validation_pending=False,validated_simulators=['mujoco'],
        gym_qualified_subset='gym_verified/manifest.json',validation_receipt='validation_receipt.json',
        amp_training_approved=False,legacy_fixed_cadence_runner_compatible=False)
    for c in m['clips']:c['validated_simulators']=['mujoco','isaac_gym'] if c['isaac_gym_verified'] else ['mujoco']
    for c in tm['clips']:
        c['isaac_gym_verified']=c['clip'] in tgood;c['validated_simulators']=['mujoco','isaac_gym'] if c['isaac_gym_verified'] else ['mujoco']
        if c['isaac_gym_verified']:c['isaac_gym_validation']=dict(receipt=str(BASE/'gym_transitions/transition_result.json'),case=tgood[c['clip']])
    tm.update(validated_simulators=['mujoco'],phase_annotation='Stored leg_phase/controller_leg_phase are commanded per-leg clocks including delay lead. V3Reference derives a common contact-aligned diagonal phase and masks phase during stand.',validation_receipt='../validation_receipt.json')
    (root/'manifest.json').write_text(json.dumps(m,indent=2)+'\n');(root/'transitions/manifest.json').write_text(json.dumps(tm,indent=2)+'\n')
    subset=root/'gym_verified';subset.mkdir(exist_ok=False);(subset/'transitions').mkdir()
    gm=copy.deepcopy(m);gm['clips']=[c for c in gm['clips'] if c['isaac_gym_verified']]
    gm.update(amp_training_approved=True,validated_simulators=['mujoco','isaac_gym'],gym_qualified_subset=None,
        validation_status='All included records passed their documented MuJoCo and Isaac Gym criteria',validation_receipt='../validation_receipt.json',
        approval_note='Physical expert subset approved. A new task is still required for stand, variable cadence, wider actor targets, and variable-command transitions. No PPO training has run.')
    for c in gm['clips']:
        os.link(root/(c['clip']+'.npz'),subset/(c['clip']+'.npz'));c['source_folder']='../'+c['source_folder']
    gt=copy.deepcopy(tm);gt['clips']=[c for c in gt['clips'] if c['isaac_gym_verified']]
    gt.update(amp_training_approved=True,validated_simulators=['mujoco','isaac_gym'],validation_receipt='../../validation_receipt.json')
    for c in gt['clips']:
        os.link(root/'transitions'/(c['clip']+'.npz'),subset/'transitions'/(c['clip']+'.npz'));c['source_folder']='../'+c['source_folder']
    (subset/'manifest.json').write_text(json.dumps(gm,indent=2)+'\n');(subset/'transitions/manifest.json').write_text(json.dumps(gt,indent=2)+'\n')
    (root/'robot').mkdir();shutil.copy2(ROOT/'unitree_rl_gym/resources/robots/urdf4/urdf/urdf4_rs01.urdf',root/'robot/urdf4_rs01.urdf');shutil.copy2(ROOT/'unitree_rl_gym/resources/robots/urdf4/BATTERY_MASS.md',root/'robot/BATTERY_MASS.md')
    scripts=['design_v2.py','design_v3.py','physical_v3.py','teacher_v3.py','export_v3.py','build_v3_release.py','reference_v3.py','validate_isaac.py','review_gym_v3.py','review_gym_transitions_v3.py','audit_v3_release.py','transitions_v3.py']
    receipt=dict(schema='urdf4_amp_v3_validation_v1',urdf_sha256=m['urdf_sha256'],total_mass_kg=m['total_mass_kg'],
        steady_clips=len(m['clips']),transition_clips=len(tm['clips']),frames=audit['frames'],
        mujoco_verified_records=len(m['clips'])+len(tm['clips']),gym_closed_loop_verified_steady=len(gm['clips']),gym_closed_loop_verified_transitions=len(gt['clips']),
        every_exported_state_matches_physical_source=True,every_frame_MuJoCo_FK=True,
        gym_static_FK=dict(passed=True,samples=fk['samples'],max_error_m=max(c['foot_fk_max_error_m'] for c in fk['cases']),receipt=str(BASE/'gym_fk_all_separated/result.json')),
        expert_interval_motion_s=[5.,55.],all_episode_fall_and_400Hz_raw_torque_gates_preserved=True,
        raw_peak_gate_Nm=17.,motor_cap_Nm=14.,max_mujoco_episode_raw_Nm=max(c['raw_peak_Nm'] for c in m['clips']+tm['clips']),
        root_or_joint_states_overwritten_during_dynamic_rollouts=False,automatic_resets=False,learned_policy_used_to_fabricate_experts=False,
        zero_command_behavior='four-foot stand; stop holds measured stopping position/heading; anchor events retained',
        old_v2_preserved=audit['old_v2_preservation'],new_RL_training_performed=False,
        independent_scope_note='Gym-qualified subset contains only passed clips. The complete dataset includes MuJoCo-only records and retains a closed legacy training guard.',
        input_manifests={str(p.relative_to(root)):digest(p) for p in [root/'manifest.json',root/'transitions/manifest.json',subset/'manifest.json',subset/'transitions/manifest.json']},
        generation_code_sha256={name:digest(ROOT/'tools/urdf4_amp'/name) for name in scripts})
    (root/'validation_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    coverage=json.loads((root/'coverage.json').read_text());coverage['gym_closed_loop_qualified_transition_clips']=len(gt['clips']);coverage['total_frames']=audit['frames'];coverage['total_minutes']=audit['frames']*.02/60
    (root/'coverage.json').write_text(json.dumps(coverage,indent=2)+'\n');print(json.dumps({k:receipt[k] for k in ['steady_clips','transition_clips','frames','gym_closed_loop_verified_steady','gym_closed_loop_verified_transitions','max_mujoco_episode_raw_Nm']},indent=2))


if __name__=='__main__':main()
