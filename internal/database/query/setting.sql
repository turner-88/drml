-- Runtime-mutable application settings.
--
-- Values are scalars stored as text and parsed by the typed accessors in
-- package store. An absent row means "no preference expressed", which the
-- caller resolves to its own default — it is not the same as a stored false.

-- name: GetSetting :one
SELECT * FROM `app_setting` WHERE setting_key = ? LIMIT 1;

-- name: ListSettings :many
SELECT * FROM `app_setting` ORDER BY setting_key;

-- name: UpsertSetting :exec
INSERT INTO `app_setting` (setting_key, setting_value, updated_at, updated_by)
VALUES (?, ?, ?, ?)
ON DUPLICATE KEY UPDATE
  setting_value = VALUES(setting_value),
  updated_at = VALUES(updated_at),
  updated_by = VALUES(updated_by);
