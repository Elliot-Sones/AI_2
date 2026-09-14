/* Thin C exports used by the frozen-player reducer parity tests. */
#include "game_player.h"

int ai2g_combat_player_size(void) { return (int)sizeof(AI2GCombatPlayer); }
int ai2g_player_physics_export(AI2NEpisode *g,AI2GCombatPlayer *e,int i,AI2NRng *rng) {
    return ai2g_player_physics(g,e,i,rng);
}
int ai2g_player_decode_export(AI2NEpisode *g,int i,const double *action) {
    return ai2_input_step_core(action,g->players[i].input_state,g->players[i].input_state);
}
void ai2g_player_set_pose_export(AI2NEpisode *g,AI2GCombatPlayer *e,int i,double x,double y,
                               double vx,double vy,int state,int facing) {
    ai2g_body_position(g,i,x,y); ai2g_body_velocity(g,i,vx,vy);
    g->players[i].ms.state_id=state; g->players[i].ms.facing=facing;
    e[i].hurtbox_x=x; e[i].hurtbox_y=y;
}
int ai2g_player_force_attack_export(AI2NEpisode *g,AI2GCombatPlayer *e,int i,int move,AI2NRng *rng) {
    const int rc=ai2g_start_attack(g,e,i,move,0);
    return rc ? rc : ai2g_transition(g,e,i,AI2G_STATE_ATTACK,rng);
}
void ai2g_player_snapshot_export(AI2NEpisode *g,AI2GCombatPlayer *e,int i,double *out) {
    const AI2NPlayer *p=&g->players[i]; const AI2MovementState *s=&p->ms;
    const AI2GAttackRuntime *a=&e[i].attack;
    const AI2CBody *b=&g->world.bodies[p->body];
    const double values[]={ai2n_body_position(b).x,ai2n_body_position(b).y,b->v.x,b->v.y,p->damage,s->stocks,s->facing,s->state_id,
        e[i].damage_taken_this_frame,e[i].damage_taken_this_stock,e[i].damage_taken_total,e[i].damage_done,
        e[i].hurtbox_x,e[i].hurtbox_y,e[i].hurtbox_width,e[i].hurtbox_height,
        ai2_mov_slot_get(s,s->state_id,AI2_MOV_SLOT_STUN_TIME_STORED,0),e[i].stun_frames,
        a->cast_idx,a->move_frame,a->total_frame_count,a->frames_into_recovery,
        a->valid ? a->cast_frame[a->cast_idx] : 0,a->hit_anyone,a->all_hit_mask,a->power_hit_mask,
        s->damage_vx,s->damage_vy,s->target_vx,s->target_vy};
    for (int k=0;k<30;++k) out[k]=values[k];
}
void ai2g_player_init_export(AI2NEpisode *scene, AI2GCombatPlayer *extra, int player) {
    ai2g_player_init(scene, extra, player);
}
void ai2g_player_pre_process_export(AI2NEpisode *scene, AI2GCombatPlayer *extra, int player) {
    ai2g_player_pre_process(scene, extra, player);
}

int ai2g_player_initialization_probe(void) {
    AI2NEpisode scene;
    AI2GCombatPlayer extra[2];
    ai2n_zero(&scene, sizeof(scene));
    scene.players[0].ms.x = 4.25;
    scene.players[0].ms.state_id = AI2_STATE_IN_AIR;
    scene.players[0].damage = 27;
    ai2g_player_init(&scene, extra, 0);
    if (scene.players[0].ms.x != 4.25 || scene.players[0].damage != 27 ||
        scene.players[0].ms.state_id != AI2_STATE_IN_AIR) return 1;
    if (extra[0].hurtbox_width != .928 || extra[0].hurtbox_height != 1.024 ||
        extra[0].hurtbox_x != 0 || extra[0].hurtbox_y != 0) return 2;
    if (extra[0].attack.valid || extra[0].attack.first_hit_agent != -1 ||
        extra[0].damage_taken_this_frame || extra[0].event_count) return 3;
    if (!scene.players[0].ms.bank[AI2G_STATE_ATTACK].present[AI2_MOV_SLOT_VALID] ||
        !scene.players[0].ms.bank[AI2G_STATE_STUN].present[AI2_MOV_SLOT_VALID] ||
        !scene.players[0].ms.bank[AI2G_STATE_TAUNT].present[AI2_MOV_SLOT_VALID]) return 4;
    return 0;
}

int ai2g_player_presence_probe(void) {
    AI2NEpisode g; AI2GCombatPlayer e[2]; AI2NRng rng;
    ai2n_zero(&g,sizeof(g)); ai2n_rng_seed(&rng,91,0);
    ai2n_movement_config(&g.mcfg);
    ai2g_player_init(&g,e,0);
    if (e[0].attribute_presence || e[0].attack.valid) return 1;
    if (ai2g_start_attack(&g,e,0,2,0)) return 2;
    if (!e[0].attack.valid || e[0].attribute_presence || !e[0].attack.in_recovery_present ||
        e[0].attack.dealt_target_components!=2 || e[0].attack.past_point_count) return 3;
    if (ai2g_transition(&g,e,0,AI2G_STATE_ATTACK,&rng)) return 4;
    if (e[0].attribute_presence!=AI2G_ATTR_ATTACK_SEED) return 5;
    if (ai2g_transition(&g,e,0,AI2G_STATE_TAUNT,&rng)) return 6;
    if (e[0].attribute_presence!=(AI2G_ATTR_ATTACK_SEED|AI2G_ATTR_TAUNT_SEED|AI2G_ATTR_TAUNT_TIMER)) return 7;
    if (!e[0].attack.valid) return 8;
    return 0;
}

int ai2g_player_event_order_probe(void) {
    AI2NEpisode g; AI2GCombatPlayer e[2];
    ai2n_zero(&g,sizeof(g)); ai2g_player_init(&g,e,0); ai2g_player_init(&g,e,1);
    ai2g_player_pre_process(&g,e,0); ai2g_player_pre_process(&g,e,1);
    if (ai2g_emit(e,0,AI2G_EVENT_ATTACK_CONTACT,1,2,0) ||
        ai2g_emit(e,0,AI2G_EVENT_DAMAGE,1,-1,3) || ai2g_emit(e,1,AI2G_EVENT_STUN,1,-1,4)) return 1;
    if (e[0].events[0].event_sequence!=0 || e[0].events[1].event_sequence!=1 ||
        e[1].events[0].event_sequence!=2 || e[1].events[0].phase_player!=1 || e[0].total_tick_events!=3) return 2;
    e[1].event_count=AI2G_PLAYER_EVENT_CAPACITY;
    if (ai2g_emit(e,1,AI2G_EVENT_STUN,1,-1,1)!=AI2G_PLAYER_ERR_EVENTS || !e[1].event_overflow) return 3;
    ai2g_player_pre_process(&g,e,0); ai2g_player_pre_process(&g,e,1);
    if (e[0].total_tick_events || e[0].event_count || e[1].event_count || e[1].event_overflow) return 4;
    return 0;
}

static int ai2g_probe_scene(AI2NEpisode *g,AI2GCombatPlayer e[2],AI2NRng *rng) {
    AI2NConfig cfg;
    ai2n_zero(&cfg,sizeof(cfg));
    cfg.max_targets=1; cfg.time_limit_steps=300; cfg.settle_steps=1; cfg.post_goal_required_steps=1;
    cfg.settle_speed=1; cfg.min_dist=2; cfg.max_dist=6; cfg.dt=1.0/30;
    ai2n_rng_seed(rng,91,0);
    int rc=ai2n_reset_scene(g,&cfg,rng);
    if (rc) return rc;
    for (int i=0;i<2;++i) {
        ai2g_player_init(g,e,i);
        ai2g_player_set_pose_export(g,e,i,3.0+i*1.1,.35,0,0,AI2_STATE_STANDING,i==0 ? 1 : -1);
    }
    return 0;
}

int ai2g_player_priority_probe(void) {
    AI2NEpisode g; AI2GCombatPlayer e[2]; AI2NRng rng; double action[10]={0};
    if (ai2g_probe_scene(&g,e,&rng)) return 1;
    action[4]=1; action[7]=1; ai2g_player_decode_export(&g,0,action);
    if (ai2g_player_physics(&g,e,0,&rng) || g.players[0].ms.state_id!=AI2_STATE_IN_AIR || e[0].attack_starts) return 2;
    if (ai2g_probe_scene(&g,e,&rng)) return 3;
    action[4]=0; action[6]=1; ai2g_player_decode_export(&g,0,action);
    if (ai2g_player_physics(&g,e,0,&rng) || g.players[0].ms.state_id!=AI2_STATE_DODGE || e[0].attack_starts) return 4;
    if (ai2g_probe_scene(&g,e,&rng)) return 5;
    action[6]=0; action[9]=1; ai2g_player_decode_export(&g,0,action);
    if (ai2g_player_physics(&g,e,0,&rng) || g.players[0].ms.state_id!=AI2G_STATE_ATTACK || e[0].attack_starts!=1) return 6;
    if (ai2g_probe_scene(&g,e,&rng)) return 7;
    ai2_mov_slot_set(&g.players[0].ms,AI2_STATE_STANDING,AI2_MOV_SLOT_STUN_TIME_STORED,4);
    ai2g_player_decode_export(&g,0,action);
    if (ai2g_player_physics(&g,e,0,&rng) || g.players[0].ms.state_id!=AI2G_STATE_STUN || e[0].stun_frames!=4 || e[0].attack_starts) return 8;
    if (ai2g_probe_scene(&g,e,&rng)) return 9;
    ai2g_body_position(&g,0,20,0);
    ai2_mov_slot_set(&g.players[0].ms,AI2_STATE_STANDING,AI2_MOV_SLOT_STUN_TIME_STORED,4);
    if (ai2g_player_physics(&g,e,0,&rng) || g.players[0].ms.state_id!=AI2_STATE_KO || g.players[0].ms.stocks!=2 || e[0].attribute_presence) return 10;
    if (ai2g_probe_scene(&g,e,&rng)) return 11;
    ai2g_player_set_pose_export(&g,e,0,3,-2,0,0,AI2G_STATE_TAUNT,1); e[0].taunt_timer=1;
    if (ai2g_player_physics(&g,e,0,&rng) || g.players[0].ms.state_id!=AI2_STATE_STANDING) return 12;
    if (ai2g_probe_scene(&g,e,&rng)) return 13;
    g.players[0].ms.state_id=AI2G_STATE_STUN; e[0].stun_frames=10;
    ai2g_body_velocity(&g,0,0,2);
    if (ai2g_player_physics(&g,e,0,&rng) || e[0].stun_frames!=9 || g.world.bodies[g.players[0].body].v.y!=-1.4) return 14;
    if (ai2g_probe_scene(&g,e,&rng)) return 15;
    action[0]=action[2]=action[3]=action[8]=1; ai2g_player_decode_export(&g,0,action);
    if (ai2g_get_move(&g,0)!=5) return 16;
    if (ai2g_probe_scene(&g,e,&rng)) return 17;
    g.players[0].ms.state_id=AI2_STATE_KO; g.players[0].damage=42; g.players[0].weapon=2;
    e[0].damage_taken_this_stock=42;
    ai2_mov_slot_set(&g.players[0].ms,AI2_STATE_KO,AI2_MOV_SLOT_KO_TIMER,1);
    if (ai2g_player_physics(&g,e,0,&rng) || g.players[0].ms.state_id!=AI2_STATE_IN_AIR || g.players[0].damage ||
        g.players[0].weapon || e[0].damage_taken_this_stock) return 18;
    if (ai2g_probe_scene(&g,e,&rng)) return 19;
    ai2g_player_set_pose_export(&g,e,0,3,-2,0,0,AI2_STATE_IN_AIR,1);
    for (int k=0;k<10;++k) action[k]=0;
    action[7]=1; ai2g_player_decode_export(&g,0,action);
    if (ai2g_player_physics(&g,e,0,&rng) || g.players[0].ms.state_id!=AI2G_STATE_ATTACK ||
        g.players[0].ms.bank[AI2G_STATE_ATTACK].present[AI2_MOV_SLOT_JUMP_TIMER]) return 20;
    return 0;
}
