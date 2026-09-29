"""Interpret guard odometry without changing the trained estimator or actor input."""
import numpy as np


def support_quality(odometry, estimator):
    """A legal, finite pair is usable even when its velocity confidence is low.

    This is kinematic evidence, not a measurement of ground contact or force.
    Confidence remains unchanged. Candidate diagnostics use the estimator's
    actual gates, and never admit a foot that the estimator rejected.
    """
    result = dict(obs_odometry_support_usable=False,
                  obs_odometry_support_state='unavailable',
                  obs_odometry_support_reason='invalid_or_missing_odometry',
                  obs_odometry_candidate_mask=[False]*4,
                  obs_odometry_leg_rejections={},
                  obs_odometry_pair_residuals_m_s=[None, None])
    try:
        confidence = float(odometry['confidence'])
        heights = np.asarray(odometry['base_height_proxy'], dtype=float).reshape(4)
        feet = np.asarray(odometry['foot_velocity'], dtype=float).reshape(4, 3)
        velocities = np.asarray(odometry['velocity_by_foot'], dtype=float).reshape(4, 3)
        vertical = np.asarray(odometry.get('support_vertical_speed', feet[:,2]), dtype=float).reshape(4)
        if not np.isfinite(np.r_[heights, feet.ravel(), velocities.ravel(), vertical, confidence]).all():
            return result
        candidates = []
        for i, leg in enumerate(estimator.LEG_ORDER):
            reasons = []
            if heights.max()-heights[i] > estimator.height_margin:
                reasons.append('height_gap')
            if abs(vertical[i]) > estimator.vertical_speed_threshold:
                reasons.append('vertical_speed')
            if abs(heights[i]-estimator.nominal_base_height) > .10:
                reasons.append('absolute_height')
            candidates.append(not reasons)
            result['obs_odometry_leg_rejections'][leg] = reasons
        result['obs_odometry_candidate_mask'] = candidates
        pairs = estimator.LEGAL_DIAGONAL_PAIRS
        residuals = [float(np.linalg.norm((velocities[a]-velocities[b])[:2])) for a,b in pairs]
        result['obs_odometry_pair_residuals_m_s'] = residuals
        index = int(odometry['selected_pair_index'])
        if not odometry['legal_diagonal_support'] or index not in range(len(pairs)):
            result['obs_odometry_support_reason'] = 'no_legal_diagonal_pair'
            return result
        pair = pairs[index]
        stance = np.asarray(odometry['stance_mask'], dtype=bool).reshape(4)
        expected = np.zeros(4, dtype=bool); expected[list(pair)] = True
        if not np.array_equal(stance, expected) or not all(candidates[i] for i in pair):
            result['obs_odometry_support_reason'] = 'invalid_selected_support'
            return result
        residual = float(odometry['pair_residual_m_s'])
        raw = np.asarray(odometry['raw_base_velocity'], dtype=float).reshape(3)
        if (not np.isfinite(np.r_[raw, residual]).all() or not 0 < confidence <= 1.
                or not 0 <= residual <= estimator.velocity_residual_threshold
                or residuals[index] > estimator.velocity_residual_threshold):
            result['obs_odometry_support_reason'] = 'unusable_pair_velocity'
            return result
        result.update(obs_odometry_support_usable=True,
                      obs_odometry_support_state='reliable' if confidence >= .5 else 'degraded',
                      obs_odometry_support_reason='' if confidence >= .5 else 'low_velocity_confidence')
    except (KeyError, TypeError, ValueError, AttributeError, IndexError):
        pass
    return result
