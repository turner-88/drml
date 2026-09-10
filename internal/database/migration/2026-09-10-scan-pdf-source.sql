-- Adds PDF report input: eye laterality and a reference to the source document.
--
-- Additive and safe to apply while the previous version is still serving, so it
-- should be run BEFORE the new binary is deployed rather than after:
--
--   * the previous release's generated SQL lists its columns explicitly, so the
--     new columns are invisible to it;
--   * source_kind has a NOT NULL DEFAULT and the other two are nullable, so its
--     inserts still satisfy every constraint.
--
-- The AFTER clauses are not cosmetic: with no migration runner, an upgraded
-- database matching a freshly created one byte for byte is what makes
-- SHOW CREATE TABLE a reliable check of whether a box is up to date.
--
-- source_ref groups the two eyes of one report; source_key is the stored report
-- itself and stays NULL unless STORAGE_KEEP_SOURCE_PDF is on.
--
-- Existing rows become source_kind = 'image' with a NULL eye, which is exactly
-- what they are. Applying this twice fails with "Duplicate column name", which
-- is the intended outcome — there is no migration runner to track state.
--
--   mysql -h <db-host> -u drml -p drml < 2026-09-10-scan-pdf-source.sql

ALTER TABLE `scan`
  ADD COLUMN `eye` varchar(2) DEFAULT NULL
      COMMENT 'OD | OS, read from the report label; NULL for a plain image upload'
      AFTER `error_message`,
  ADD COLUMN `source_kind` varchar(8) NOT NULL DEFAULT 'image'
      COMMENT 'image | pdf'
      AFTER `eye`,
  ADD COLUMN `source_ref` char(32) DEFAULT NULL
      COMMENT 'groups the eyes of one report; not a storage key'
      AFTER `source_kind`,
  ADD COLUMN `source_key` varchar(255) DEFAULT NULL
      COMMENT 'storage key of the retained source PDF, when STORAGE_KEEP_SOURCE_PDF'
      AFTER `source_ref`,
  ADD KEY `idx_scan_source_ref` (`source_ref`);
