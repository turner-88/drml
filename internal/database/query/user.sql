-- name: GetUserByID :one
SELECT * FROM `user` WHERE id = ? LIMIT 1;

-- name: GetUserByUsername :one
SELECT * FROM `user` WHERE username = ? LIMIT 1;

-- name: GetUserByEmail :one
SELECT * FROM `user` WHERE email = ? LIMIT 1;

-- name: CreateUser :execresult
INSERT INTO `user` (
  name, email, username, password_hash, role, subrole,
  must_change_password, created_at, updated_at, created_by, updated_by
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);

-- name: UpdateUser :exec
UPDATE `user`
SET name = ?, email = ?, username = ?, role = ?, updated_at = ?, updated_by = ?
WHERE id = ?;

-- name: UpdateUserPassword :exec
UPDATE `user`
SET password_hash = ?, must_change_password = NULL, updated_at = ?, updated_by = ?
WHERE id = ?;

-- ResetUserPassword only matches while the hash is still the one the reset link
-- was issued against, so two submissions of the same link cannot both succeed.
-- name: ResetUserPassword :execrows
UPDATE `user`
SET password_hash = sqlc.arg(new_hash), must_change_password = NULL,
    updated_at = sqlc.arg(updated_at), updated_by = sqlc.arg(updated_by)
WHERE id = sqlc.arg(id) AND password_hash = sqlc.arg(old_hash);

-- AdminSetUserPassword flags the account like CreateUser does: the password was
-- chosen by an administrator, not by the account holder.
-- name: AdminSetUserPassword :exec
UPDATE `user`
SET password_hash = ?, must_change_password = 1, updated_at = ?, updated_by = ?
WHERE id = ?;

-- name: SetUserSuspended :exec
UPDATE `user` SET suspended_at = ?, updated_at = ?, updated_by = ? WHERE id = ?;

-- DeleteUser refuses an account that owns scans. fk_scan_created_by would
-- otherwise null out their created_by, silently orphaning clinical records.
-- name: DeleteUser :execrows
DELETE FROM `user`
WHERE `user`.id = sqlc.arg(id)
  AND NOT EXISTS (SELECT 1 FROM `scan` s WHERE s.created_by = `user`.id);

-- name: ListUsers :many
-- scan_count decides whether the account may be deleted; idx_scan_created_by
-- keeps the subquery to an index range per listed row.
SELECT u.*, (SELECT COUNT(*) FROM `scan` s WHERE s.created_by = u.id) AS scan_count
FROM `user` u
WHERE (sqlc.narg(role) IS NULL OR u.role = sqlc.narg(role))
ORDER BY u.id DESC
LIMIT ? OFFSET ?;

-- name: CountUsers :one
SELECT COUNT(*) FROM `user`
WHERE (sqlc.narg(role) IS NULL OR role = sqlc.narg(role));

-- name: CountUsersByRole :one
SELECT COUNT(*) FROM `user` WHERE role = ?;
