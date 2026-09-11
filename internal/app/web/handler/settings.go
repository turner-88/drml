package handler

import (
	"log"
	"net/http"

	"github.com/remorac/drml/internal/database/store"
)

// SettingsPage renders the administrator settings form.
func (h *Handler) SettingsPage(w http.ResponseWriter, r *http.Request) {
	h.render(w, r, "settings", h.settingsData(r, ""))
}

// settingsData builds the page's view model. Shared with UpdateSettings so an
// error re-render shows the same live state as a fresh load.
func (h *Handler) settingsData(r *http.Request, errMsg string) map[string]any {
	data := map[string]any{
		"Title": "Pengaturan",
		// Saved drives the success notice. There is no flash infrastructure in
		// this app, so the redirect carries it in the query string — the same
		// idiom the login page uses for ?blocked=1.
		"Saved":               r.URL.Query().Get("saved") == "1",
		"Calibration":         h.engine.GateCalibration(),
		"RegistrationEnabled": h.registrationOpen(r.Context()),
	}
	if errMsg != "" {
		data["Error"] = errMsg
	}
	return data
}

// UpdateSettings applies the administrator's gate and registration preferences
// from the single settings form.
//
// With gate.json absent the gate checkbox is disabled and submits nothing, so
// the gate is left untouched rather than read as off: otherwise every
// registration change would silently store gate=false.
func (h *Handler) UpdateSettings(w http.ResponseWriter, r *http.Request) {
	if err := r.ParseForm(); err != nil {
		h.render(w, r, "settings", h.settingsData(r, "Gagal memproses formulir pengaturan."))
		return
	}

	// An unchecked checkbox submits nothing, so absence means off.
	wantGate := r.FormValue("gate_enabled") == "on"
	wantRegistration := r.FormValue("registration_enabled") == "on"

	me := actor(r)
	ctx := r.Context()

	if h.engine.GateAvailable() {
		wasGate := h.engine.GateEnabled()

		// Persist before flipping the in-memory flag. The other order would leave a
		// running process disagreeing with the database whenever the write fails,
		// and the disagreement would only surface at the next restart.
		if err := h.store.SetBoolSetting(ctx, store.SettingGateEnabled, wantGate, me.ID); err != nil {
			log.Printf("settings: save gate.enabled: %v", err)
			h.render(w, r, "settings", h.settingsData(r, "Gagal menyimpan perubahan pengaturan."))
			return
		}

		if !h.engine.SetGateEnabled(wantGate) {
			// Only reachable if the calibration vanished between the check above
			// and here; the stored value is already correct for the next restart.
			log.Println("settings: gate enable refused, no calibration loaded")
			h.render(w, r, "settings", h.settingsData(r,
				"Pengaturan berhasil disimpan, namun filter tidak dapat diaktifkan karena file model/gate.json belum tersedia."))
			return
		}

		// Disabling a clinical safety control is worth an audit line of its own,
		// beyond the updated_by column on the row. Only on change, since every
		// save of the shared form rewrites both settings.
		if wantGate != wasGate {
			log.Printf("settings: non-fundus gate set to %v by %s (id %d)", wantGate, me.Username, me.ID)
		}
	}

	wasRegistration := h.registrationOpen(ctx)
	if err := h.store.SetBoolSetting(ctx, store.SettingRegistrationEnabled, wantRegistration, me.ID); err != nil {
		log.Printf("settings: save registration.enabled: %v", err)
		h.render(w, r, "settings", h.settingsData(r, "Gagal menyimpan perubahan pengaturan."))
		return
	}
	if wantRegistration != wasRegistration {
		log.Printf("settings: public registration set to %v by %s (id %d)", wantRegistration, me.Username, me.ID)
	}

	http.Redirect(w, r, "/admin/settings?saved=1", http.StatusSeeOther)
}
