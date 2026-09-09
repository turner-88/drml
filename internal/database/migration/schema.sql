-- DRML — Diabetic Retinopathy screening
-- Conventions follow the existing Go monolith: int(11) unix-second timestamps,
-- created_by/updated_by actor FKs, utf8mb4.

SET NAMES utf8mb4;
SET time_zone = '+00:00';
SET foreign_key_checks = 0;
SET sql_mode = 'NO_AUTO_VALUE_ON_ZERO';

DROP TABLE IF EXISTS `user`;
CREATE TABLE `user` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `name` varchar(100) DEFAULT NULL,
  `email` varchar(100) DEFAULT NULL,
  `username` varchar(50) NOT NULL COMMENT 'letter: required, number: optional, symbol: forbidden',
  `password_hash` varchar(255) NOT NULL,
  `role` int(11) NOT NULL DEFAULT 1 COMMENT '1 = clinician, 2 = administrator',
  `subrole` int(11) NOT NULL DEFAULT 1 COMMENT '1 = staff, 9 = superuser',
  `must_change_password` int(11) DEFAULT NULL,
  `suspended_at` int(11) DEFAULT NULL,
  `created_at` int(11) DEFAULT NULL,
  `updated_at` int(11) DEFAULT NULL,
  `created_by` int(11) DEFAULT NULL,
  `updated_by` int(11) DEFAULT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `uq_user_username` (`username`),
  KEY `idx_user_role` (`role`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

DROP TABLE IF EXISTS `scan`;
CREATE TABLE `scan` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `patient_ref` varchar(64) DEFAULT NULL COMMENT 'optional clinic-side patient identifier',
  `notes` text DEFAULT NULL,
  `image_key` varchar(255) NOT NULL COMMENT 'storage key, not a URL — resolved via storage.URL()',
  `heatmap_key` varchar(255) DEFAULT NULL COMMENT 'attention-rollout overlay',
  `image_sha256` char(64) NOT NULL COMMENT 'dedupe + integrity',
  `predicted_grade` tinyint(4) DEFAULT NULL COMMENT '0..4 APTOS DR severity',
  `confidence` float DEFAULT NULL COMMENT 'top-1 softmax probability',
  `probabilities` json DEFAULT NULL COMMENT '[p0,p1,p2,p3,p4]',
  `model_id` varchar(128) NOT NULL COMMENT 'provenance: predictions are model-specific',
  `model_sha256` char(64) NOT NULL,
  `inference_ms` int(11) DEFAULT NULL,
  `status` varchar(16) NOT NULL DEFAULT 'done' COMMENT 'done | failed',
  `error_message` varchar(255) DEFAULT NULL,
  `created_at` int(11) DEFAULT NULL,
  `created_by` int(11) DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `idx_scan_created_at` (`created_at`),
  KEY `idx_scan_grade_created` (`predicted_grade`, `created_at`),
  KEY `idx_scan_patient` (`patient_ref`),
  KEY `idx_scan_created_by` (`created_by`, `created_at`),
  CONSTRAINT `fk_scan_created_by` FOREIGN KEY (`created_by`) REFERENCES `user` (`id`) ON DELETE SET NULL ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

DROP TABLE IF EXISTS `app_setting`;
CREATE TABLE `app_setting` (
  `setting_key` varchar(64) NOT NULL,
  `setting_value` varchar(255) NOT NULL COMMENT 'scalar; parsed by the typed accessor in store',
  `updated_at` int(11) DEFAULT NULL,
  `updated_by` int(11) DEFAULT NULL,
  PRIMARY KEY (`setting_key`),
  KEY `idx_app_setting_updated_by` (`updated_by`),
  CONSTRAINT `fk_app_setting_updated_by` FOREIGN KEY (`updated_by`) REFERENCES `user` (`id`) ON DELETE SET NULL ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

SET foreign_key_checks = 1;
