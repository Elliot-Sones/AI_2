#ifndef AI2_GAME_PLAYER_TYPES_H
#define AI2_GAME_PLAYER_TYPES_H

#include <stdint.h>

/* Sidecar state: the existing AI2NEpisode/MovementState ABI is unchanged. */
#define AI2G_STATE_STUN 5
#define AI2G_STATE_ATTACK 8
#define AI2G_STATE_TAUNT 12
#define AI2G_ATTACK_MAX_CASTS 9
#define AI2G_PLAYER_EVENT_CAPACITY 32
#define AI2G_PLAYER_ERR_INVALID 101
#define AI2G_PLAYER_ERR_ATTACK_DATA 102
#define AI2G_PLAYER_ERR_RNG 103
#define AI2G_PLAYER_ERR_EVENTS 104
#define AI2G_PLAYER_ERR_SOURCE_EXCEPTION 105
#define AI2G_ATTR_STUN_FRAMES 1u
#define AI2G_ATTR_TAUNT_TIMER 2u
#define AI2G_ATTR_TAUNT_SEED 4u
#define AI2G_ATTR_ATTACK_SEED 8u
#define AI2G_ATTR_GRAVITY_SCALE 16u

enum {
    AI2G_EVENT_ATTACK_STARTED = 1,
    AI2G_EVENT_ATTACK_ACTIVE = 2,
    AI2G_EVENT_ATTACK_CONTACT = 3,
    AI2G_EVENT_DAMAGE = 4,
    AI2G_EVENT_ATTACK_ENDED = 5,
    AI2G_EVENT_STUN = 6,
    AI2G_EVENT_KO = 7,
    AI2G_EVENT_RESPAWN = 8,
    AI2G_EVENT_POWER_CHANGED = 9
};

typedef struct {
    int32_t kind, source, target, move, power, cast, hitbox, phase_player, event_sequence;
    uint64_t attack_id;
    double amount;
} AI2GPlayerEvent;

typedef struct {
    int32_t move_idx, power_idx, cast_idx, move_frame, total_frame_count;
    int32_t frames_into_recovery, cast_frame[AI2G_ATTACK_MAX_CASTS];
    int32_t cooldown, move_facing, first_hit_agent;
    uint32_t all_hit_mask, power_hit_mask;
    int32_t valid, hit_anyone, is_switching_casts, in_recovery, dealt_target_exists;
    int32_t in_recovery_present, dealt_target_components, past_point_count;
    double dealt_target_x, dealt_target_y, dealt_target_mult;
    double past_point_x, past_point_y;
    uint64_t attack_id;
} AI2GAttackRuntime;

typedef struct {
    double damage_taken_this_stock, damage_taken_total, damage_taken_this_frame, damage_done;
    double hurtbox_x, hurtbox_y, hurtbox_width, hurtbox_height;
    /* Python writes these attributes, but CapsuleCollider never consumes them. */
    double hurtbox_offset_x, hurtbox_offset_y, hurtbox_size_x, hurtbox_size_y;
    int32_t hurtbox_attributes_present, gravity_scale, just_got_hit;
    int32_t stun_frames, taunt_timer, taunt_seed, attack_seed;
    uint32_t attribute_presence;
    AI2GAttackRuntime attack;
    uint64_t attack_starts, attack_contacts, damage_events, hit_during_stun;
    int32_t last_damage_source;
    uint64_t last_damage_attack_id;
    int32_t event_count, event_overflow, total_tick_events;
    AI2GPlayerEvent events[AI2G_PLAYER_EVENT_CAPACITY];
} AI2GCombatPlayer;

#endif
