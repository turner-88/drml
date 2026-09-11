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
SET name = ?, email = ?, role = ?, subrole = ?, updated_at = ?, updated_by = ?
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

-- name: SetUserSuspended :exec
UPDATE `user` SET suspended_at = ?, updated_at = ?, updated_by = ? WHERE id = ?;

-- name: DeleteUser :exec
DELETE FROM `user` WHERE id = ?;

-- name: ListUsers :many
SELECT * FROM `user`
WHERE (sqlc.narg(role) IS NULL OR role = sqlc.narg(role))
ORDER BY id DESC
LIMIT ? OFFSET ?;

-- name: CountUsers :one
SELECT COUNT(*) FROM `user`
WHERE (sqlc.narg(role) IS NULL OR role = sqlc.narg(role));

-- name: CountUsersByRole :one
SELECT COUNT(*) FROM `user` WHERE role = ?;
