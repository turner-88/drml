-- Analytics aggregate in SQL, never in Go: the scan table's indexes
-- (idx_scan_grade_created, idx_scan_created_by) exist precisely so these stay
-- cheap as the archive grows on a 2 GB box.

-- name: GradeDistribution :many
SELECT predicted_grade, COUNT(*) AS total
FROM `scan`
WHERE status = 'done'
  AND predicted_grade IS NOT NULL
  AND (sqlc.arg(all_scans) = 1 OR created_by = sqlc.arg(created_by))
  AND (sqlc.narg(from_ts) IS NULL OR created_at >= sqlc.narg(from_ts))
  AND (sqlc.narg(to_ts) IS NULL OR created_at <= sqlc.narg(to_ts))
GROUP BY predicted_grade
ORDER BY predicted_grade;

-- name: ScansPerDay :many
SELECT
  FROM_UNIXTIME(created_at, '%Y-%m-%d') AS day,
  COUNT(*) AS total,
  SUM(predicted_grade >= 2) AS referable
FROM `scan`
WHERE status = 'done'
  AND (sqlc.arg(all_scans) = 1 OR created_by = sqlc.arg(created_by))
  AND (sqlc.narg(from_ts) IS NULL OR created_at >= sqlc.narg(from_ts))
  AND (sqlc.narg(to_ts) IS NULL OR created_at <= sqlc.narg(to_ts))
GROUP BY day
ORDER BY day;

-- name: ConfidenceByGrade :many
SELECT
  predicted_grade,
  AVG(confidence) AS mean_confidence,
  MIN(confidence) AS min_confidence,
  COUNT(*) AS total
FROM `scan`
WHERE status = 'done'
  AND predicted_grade IS NOT NULL
  AND (sqlc.arg(all_scans) = 1 OR created_by = sqlc.arg(created_by))
  AND (sqlc.narg(from_ts) IS NULL OR created_at >= sqlc.narg(from_ts))
  AND (sqlc.narg(to_ts) IS NULL OR created_at <= sqlc.narg(to_ts))
GROUP BY predicted_grade
ORDER BY predicted_grade;

-- name: SummaryStats :one
-- Referable DR (grade >= 2) is the clinically actionable threshold — the rate
-- that decides whether a patient needs an ophthalmologist referral.
SELECT
  COUNT(*) AS total_scans,
  COALESCE(SUM(predicted_grade >= 2), 0) AS referable_count,
  COALESCE(AVG(confidence), 0) AS mean_confidence,
  COALESCE(SUM(confidence < sqlc.arg(low_confidence)), 0) AS low_confidence_count,
  COALESCE(AVG(inference_ms), 0) AS mean_inference_ms
FROM `scan`
WHERE status = 'done'
  AND (sqlc.arg(all_scans) = 1 OR created_by = sqlc.arg(created_by))
  AND (sqlc.narg(from_ts) IS NULL OR created_at >= sqlc.narg(from_ts))
  AND (sqlc.narg(to_ts) IS NULL OR created_at <= sqlc.narg(to_ts));

-- name: LowConfidenceScans :many
-- At ~70% accuracy the model's own uncertainty is the most useful signal it
-- produces; this queue is what makes the tool honest rather than oracular.
SELECT * FROM `scan`
WHERE status = 'done'
  AND confidence < sqlc.arg(low_confidence)
  AND (sqlc.arg(all_scans) = 1 OR created_by = sqlc.arg(created_by))
  AND (sqlc.narg(from_ts) IS NULL OR created_at >= sqlc.narg(from_ts))
  AND (sqlc.narg(to_ts) IS NULL OR created_at <= sqlc.narg(to_ts))
ORDER BY confidence ASC, created_at DESC
LIMIT ?;
