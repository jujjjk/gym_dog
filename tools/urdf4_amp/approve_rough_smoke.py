"""Bind the rough training launch to successful rollouts of the same weights."""
import argparse
import hashlib
import json
from pathlib import Path


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', type=Path, required=True)
    parser.add_argument('--rough-result', type=Path, required=True)
    parser.add_argument('--flat-result', type=Path, required=True)
    args = parser.parse_args()
    integration = json.loads((args.smoke / 'smoke.json').read_text())
    rough = json.loads(args.rough_result.read_text())
    flat = json.loads(args.flat_result.read_text())
    iteration = integration['iteration']
    checkpoint = (args.smoke / f'model_{iteration}.pt').resolve()
    student = args.smoke / 'exported' / f'student_{iteration}.pt'
    if not integration['passed'] or integration['push_count'] <= 0:
        raise ValueError('Training integration and disturbance smoke did not pass')
    for result in (rough, flat):
        if (result['seconds'] < 30 or len(result['cases']) != 22 or result['copies'] < 2
                or not result['core_passed'] or not result['tested_subset_passed']
                or result['automatic_resets'] or result['teacher_controller_used']):
            raise ValueError('All 22 requested motions must pass the 30-second public-policy smoke')
    if (not rough['rough_ground'] or rough['rough_fraction'] != 1.
            or rough['checkpoint_sha256'] != file_sha(checkpoint)
            or flat['public_policy_sha256'] != file_sha(student)
            or max(abs(value) for value in rough['terrain_height_bounds_m']) < .001):
        raise ValueError('Evaluation checkpoint or rough-ground conditions differ')
    receipt = dict(approved_for_training=True, rough_behavior_passed=True, flat_regression_passed=True,
                   checkpoint=str(checkpoint), checkpoint_sha256=file_sha(checkpoint),
                   source_sha256=rough['source_sha256'], contract=rough['contract'],
                   integration=str((args.smoke / 'smoke.json').resolve()),
                   rough_result=str(args.rough_result.resolve()), flat_result=str(args.flat_result.resolve()))
    output = Path(__file__).resolve().parents[2] / 'artifacts/urdf4_rough_20261118/training_receipt.json'
    with output.open('x') as file:
        json.dump(receipt, file, indent=2)
        file.write('\n')
    print(output)


if __name__ == '__main__':
    main()
