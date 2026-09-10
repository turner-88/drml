-- name: CreateScan :execresult
INSERT INTO `scan` (
  patient_ref, notes, image_key, heatmap_key, image_sha256,
  predicted_grade, confidence, probabilities,
  model_id, model_sha256, inference_ms,
  status, error_message, eye, source_kind, source_ref, source_key, created_at, created_by
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);

-- name: GetScan :one
SELECT * FROM `scan` WHERE id = ? LIMIT 1;

-- name: GetScanForUser :one
-- Role scoping lives in the query, not in Go: a clinician may only read their
-- own scans, and passing sqlc.arg(all_scans) = 1 for admins keeps the two
-- paths on a single statement so they cannot drift apart.
SELECT * FROM `scan`
WHERE id = ?
  AND (sqlc.arg(all_scans) = 1 OR created_by = sqlc.arg(created_by))
LIMIT 1;

-- name: DeleteScan :exec
DELETE FROM `scan`
WHERE id = ?
  AND (sqlc.arg(all_scans) = 1 OR created_by = sqlc.arg(created_by));

-- name: ListScans :many
SELECT * FROM `scan`
WHERE (sqlc.arg(all_scans) = 1 OR created_by = sqlc.arg(created_by))
  AND (sqlc.narg(grade) IS NULL OR predicted_grade = sqlc.narg(grade))
  AND (sqlc.narg(patient_ref) IS NULL OR patient_ref = sqlc.narg(patient_ref))
  AND (sqlc.narg(from_ts) IS NULL OR created_at >= sqlc.narg(from_ts))
  AND (sqlc.narg(to_ts) IS NULL OR created_at <= sqlc.narg(to_ts))
ORDER BY created_at DESC, id DESC
LIMIT ? OFFSET ?;

-- name: CountScans :one
SELECT COUNT(*) FROM `scan`
WHERE (sqlc.arg(all_scans) = 1 OR created_by = sqlc.arg(created_by))
  AND (sqlc.narg(grade) IS NULL OR predicted_grade = sqlc.narg(grade))
  AND (sqlc.narg(patient_ref) IS NULL OR patient_ref = sqlc.narg(patient_ref))
  AND (sqlc.narg(from_ts) IS NULL OR created_at >= sqlc.narg(from_ts))
  AND (sqlc.narg(to_ts) IS NULL OR created_at <= sqlc.narg(to_ts));

-- name: RecentScans :many
SELECT * FROM `scan`
WHERE (sqlc.arg(all_scans) = 1 OR created_by = sqlc.arg(created_by))
  AND status = 'done'
ORDER BY created_at DESC, id DESC
LIMIT ?;

-- name: ListScansBySourceRef :many
-- Both eyes of one PDF report share a source_ref, which is what lets the detail
-- page offer the other eye without a separate exam table. It is a plain grouping
-- token, so it works whether or not the report itself was retained.
SELECT * FROM `scan`
WHERE source_ref = sqlc.arg(source_ref)
  AND (sqlc.arg(all_scans) = 1 OR created_by = sqlc.arg(created_by))
ORDER BY eye, id;
