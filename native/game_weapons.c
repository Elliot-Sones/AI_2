/* Thin C ABI for local differential tests and capture adapters. */
#include "game_weapons.h"

int ai2g_weapons_state_size(void) { return (int)sizeof(AI2GWeapons); }
int ai2g_weapons_aux_size(void) { return (int)sizeof(AI2GWeaponPlayerAux); }
int ai2g_weapons_weapon_size(void) { return (int)sizeof(AI2GWeapon); }
int ai2g_weapons_spawner_size(void) { return (int)sizeof(AI2GSpawner); }
int ai2g_weapons_init_export(AI2NEpisode *scene, AI2GWeapons *s, AI2GWeaponPlayerAux *aux, double scale) {
    return ai2g_weapons_init_from_scene(scene, s, aux, scale);
}
int ai2g_weapons_mirror_export(AI2NEpisode *scene, AI2GWeapons *s) {
    return ai2g_weapons_mirror_before_physics(scene, s);
}
int ai2g_weapons_drop_export(AI2NEpisode *scene, AI2GWeapons *s, AI2GWeaponPlayerAux *aux, AI2NRng *rng) {
    return ai2g_weapons_drop_all(scene, s, aux, rng);
}
int ai2g_weapons_pickup_export(AI2NEpisode *scene, AI2GWeapons *s, AI2GWeaponPlayerAux *aux, AI2NRng *rng) {
    return ai2g_weapons_pickup_all(scene, s, aux, rng);
}
int ai2g_weapons_update_export(AI2NEpisode *scene, AI2GWeapons *s, AI2NRng *rng) {
    return ai2g_weapons_update_and_prune(scene, s, rng);
}
