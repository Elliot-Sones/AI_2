#ifndef AI2_GAME_PLAYER_H
#define AI2_GAME_PLAYER_H

#include "navigation_episode.h"
#include "game_player_types.h"
#include "game_attack_data.h"
#include "game_capsule.h"

#ifndef AI2G_HD
#define AI2G_HD AI2N_HD
#endif

AI2G_HD static inline void ai2g_player_init(AI2NEpisode *scene, AI2GCombatPlayer extra[2], int player) {
    AI2GCombatPlayer *e = &extra[player];
    ai2n_zero(e, sizeof(*e));
    e->hurtbox_width = (290 / 2) * 2 * (1.024 / 320);
    e->hurtbox_height = (320 / 2) * 2 * (1.024 / 320);
    e->gravity_scale = 1;
    e->last_damage_source = -1;
    e->attack.move_idx = -1;
    e->attack.power_idx = -1;
    e->attack.first_hit_agent = -1;
    for (int state = 0; state < AI2_MOVEMENT_STATE_COUNT; ++state)
        ai2_mov_ensure_slot(&scene->players[player].ms, state);
}

/* Called for BOTH players before either player's state physics. */
AI2G_HD static inline void ai2g_player_pre_process(AI2NEpisode *scene, AI2GCombatPlayer extra[2], int player) {
    (void)scene;
    extra[player].damage_taken_this_frame = 0.0;
    extra[player].event_count = 0;
    extra[player].event_overflow = 0;
    if (player==0) extra[0].total_tick_events=0;
}

AI2G_HD static inline int ai2g_emit(AI2GCombatPlayer extra[2], int source, int kind,
                                   int target, int hitbox, double amount) {
    if (!extra || source<0 || source>1) return AI2G_PLAYER_ERR_INVALID;
    AI2GCombatPlayer *e = &extra[source];
    if (e->event_count<0 || e->event_count>=AI2G_PLAYER_EVENT_CAPACITY ||
        extra[0].total_tick_events<0 || extra[0].total_tick_events==INT32_MAX) {
        e->event_overflow = 1;
        return AI2G_PLAYER_ERR_EVENTS;
    }
    AI2GPlayerEvent *v = &e->events[e->event_count++];
    v->kind=kind; v->source=source; v->target=target; v->hitbox=hitbox; v->amount=amount;
    v->phase_player=source; v->event_sequence=extra[0].total_tick_events++;
    v->move=e->attack.move_idx; v->power=e->attack.power_idx; v->cast=e->attack.cast_idx;
    v->attack_id=e->attack.attack_id;
    return 0;
}

AI2G_HD static inline void ai2g_body_velocity(AI2NEpisode *g, int i, double x, double y) {
    g->world.bodies[g->players[i].body].v = ai2c_v(x,y);
    g->players[i].ms.vx=x; g->players[i].ms.vy=y;
}

AI2G_HD static inline void ai2g_body_position(AI2NEpisode *g, int i, double x, double y) {
    AI2CBody *b=&g->world.bodies[g->players[i].body];
    b->p=ai2c_v(x,y); ai2c_body_set_transform(b,b->p,b->a);
    g->players[i].ms.x=x; g->players[i].ms.y=y;
}

AI2G_HD static inline double ai2g_units(double x) { return x * 2 * (1.024 / 320); }
AI2G_HD static inline double ai2g_radians(double x) { return x * (3.14159265358979323846 / 180.0); }

/* Player.get_move: held heavy takes priority; W takes precedence over down/side. */
AI2G_HD static inline int ai2g_get_move(AI2NEpisode *g, int i) {
    const int32_t *in=g->players[i].input_state;
    const int heavy=in[8], light=!heavy && in[7];
    if (!heavy && !light) return 1;
    const int side=in[1] || in[3];
    const int neutral=(!side && !in[2]) || in[0];
    const int direction=neutral ? 0 : (in[2] ? 1 : 2);
    if (ai2n_is_on_floor(g,i)) return (heavy ? 5 : 2)+direction;
    return heavy ? (direction==1 ? 12 : 11) : 8+direction;
}

AI2G_HD static inline int ai2g_power_begin(AI2GAttackRuntime *a, int power) {
    if (power<0 || power>=AI2G_POWER_COUNT || ai2g_powers[power].cast_count<1 ||
        ai2g_powers[power].cast_count>AI2G_ATTACK_MAX_CASTS) return AI2G_PLAYER_ERR_ATTACK_DATA;
    a->power_idx=power; a->cast_idx=0; a->total_frame_count=0; a->frames_into_recovery=0;
    a->power_hit_mask=0; a->hit_anyone=0; a->is_switching_casts=1; a->in_recovery=0;
    a->in_recovery_present=1; a->dealt_target_components=2; a->past_point_count=0;
    a->dealt_target_exists=0; a->dealt_target_x=0; a->dealt_target_y=0; a->dealt_target_mult=0;
    a->past_point_x=0; a->past_point_y=0;
    for (int k=0;k<AI2G_ATTACK_MAX_CASTS;++k) a->cast_frame[k]=0;
    return 0;
}

/* AttackState.give_move runs before the state transition/enter callback. */
AI2G_HD static inline int ai2g_attack_begin(AI2NEpisode *g, AI2GCombatPlayer e[2], int i, int move) {
    const int weapon=g->players[i].weapon;
    if (move<2 || move>12 || weapon<0 || weapon>2) return AI2G_PLAYER_ERR_ATTACK_DATA;
    AI2GAttackRuntime *a=&e[i].attack;
    ai2n_zero(a,sizeof(*a));
    a->move_idx=weapon*11+move-2; a->first_hit_agent=-1; a->move_facing=g->players[i].ms.facing;
    a->valid=1; a->attack_id=++e[i].attack_starts;
    const int rc=ai2g_power_begin(a,ai2g_moves[a->move_idx].initial_power);
    if (rc) return rc;
    a->cooldown=ai2g_powers[a->power_idx].cooldown;
    return ai2g_emit(e,i,AI2G_EVENT_ATTACK_STARTED,-1,-1,0);
}

AI2G_HD static inline int ai2g_apply_damage(AI2NEpisode *g, AI2GCombatPlayer e[2], int source,
                                           int target, double amount, int stun, double fx, double fy) {
    AI2NPlayer *p=&g->players[target];
    p->damage=ai2n_pymin(700,p->damage+amount);
    e[target].damage_taken_this_stock+=amount; e[target].damage_taken_total+=amount;
    e[target].damage_taken_this_frame+=amount;
    ai2_mov_slot_set(&p->ms,p->ms.state_id,AI2_MOV_SLOT_STUN_TIME_STORED,stun);
    if (ai2n_is_on_floor(g,target) && fy>0) fy=-fy*.7;
    p->ms.damage_vx=fx*((1.024/320.0)*30);
    p->ms.damage_vy=fy*((1.024/320.0)*30);
    e[source].damage_done+=amount; e[source].damage_events++;
    e[target].last_damage_source=source; e[target].last_damage_attack_id=e[source].attack.attack_id;
    return ai2g_emit(e,source,AI2G_EVENT_DAMAGE,target,-1,amount);
}

AI2G_HD static inline void ai2g_xy_velocity(AI2NEpisode *g,int target,const AI2GXYChange *v,int facing,int add) {
    AI2CBody *b=&g->world.bodies[g->players[target].body];
    double x=b->v.x,y=b->v.y;
    if (v->active_x) x=add ? x+v->x*facing : v->x*facing;
    if (v->active_y) y=add ? y+v->y : v->y;
    ai2g_body_velocity(g,target,x,y);
}

AI2G_HD static inline void ai2g_frame_changes(AI2NEpisode *g,AI2GCombatPlayer e[2],int i,
                                             const AI2GFrameChange *f,int floor_drag) {
    AI2CBody *body=&g->world.bodies[g->players[i].body];
    AI2GAttackRuntime *a=&e[i].attack;
    if (floor_drag) ai2g_body_velocity(g,i,ai2_mov_move_toward(body->v.x,0,g->mcfg.in_air_ease),body->v.y);
    if (!f) return;
    if (f->hurtbox_active) {
        e[i].hurtbox_attributes_present=1;
        e[i].hurtbox_offset_x=ai2g_units(f->hurtbox_x)*a->move_facing;
        e[i].hurtbox_offset_y=ai2g_units(f->hurtbox_y);
        e[i].hurtbox_size_x=2.0*ai2g_units(f->hurtbox_width);
        e[i].hurtbox_size_y=2.0*ai2g_units(f->hurtbox_height);
    }
    /* casterPositionChange and the hurtbox .offset/.size above are source no-ops. */
    if (a->dealt_target_exists) {
        const double tx=ai2g_units(a->dealt_target_x)*a->move_facing;
        const double ty=ai2g_units(a->dealt_target_y);
        const uint32_t mask=(ai2g_powers[a->power_idx].flags&8) ? a->all_hit_mask :
                            (a->first_hit_agent<0 ? 0u : 1u<<a->first_hit_agent);
        for (int j=0;j<2;++j) if (mask&(1u<<j)) {
            const AI2CBody *target=&g->world.bodies[g->players[j].body];
            g->players[j].ms.target_vx=a->dealt_target_mult*((ai2n_body_position(body).x+tx)-ai2n_body_position(target).x);
            g->players[j].ms.target_vy=a->dealt_target_mult*((ai2n_body_position(body).y+ty)-ai2n_body_position(target).y);
        }
    }
    if (f->caster_velocity_active) {
        const double angle=ai2g_radians(f->caster_angle);
        ai2g_body_velocity(g,i,(cos(angle)*f->caster_speed)*a->move_facing,-sin(angle)*f->caster_speed);
    }
    double x=body->v.x,y=body->v.y;
    if (f->caster_damp.active_x) x*=f->caster_damp.x;
    if (f->caster_damp.active_y) y*=f->caster_damp.y;
    ai2g_body_velocity(g,i,x,y);
    ai2g_xy_velocity(g,i,&f->caster_set,a->move_facing,0);
    ai2g_xy_velocity(g,i,&f->caster_add,a->move_facing,1);
    for (int j=0;j<2;++j) if (a->all_hit_mask&(1u<<j)) ai2g_xy_velocity(g,j,&f->hit_set,a->move_facing,0);
    for (int j=0;j<2;++j) if (a->all_hit_mask&(1u<<j)) ai2g_xy_velocity(g,j,&f->hit_add,a->move_facing,1);
    for (int j=0;j<2;++j) if (a->all_hit_mask&(1u<<j)) {
        const AI2CBody *target=&g->world.bodies[g->players[j].body];
        x=ai2n_body_position(target).x; y=ai2n_body_position(target).y;
        if (f->hit_position.active_x) x=ai2n_body_position(body).x+ai2g_units(f->hit_position.x)*a->move_facing;
        if (f->hit_position.active_y) y=ai2n_body_position(body).y+ai2g_units(f->hit_position.y);
        if (f->hit_position.active_x || f->hit_position.active_y) ai2g_body_position(g,j,x,y);
    }
}

AI2G_HD static inline double ai2g_force(const AI2GCastData *cast,double damage) {
    const double x=damage+cast->damage;
    return cast->fixed_force+cast->variable_force*(x/70+AI2N_POW(x,2.0)/12000);
}

/* Validate dynamic table/agent indices before any lookup, shift or state write.
 * The immutable generated tables themselves are verified by the asset oracle. */
AI2G_HD static inline int ai2g_attack_indices_valid(const AI2GAttackRuntime *a) {
    if (!a->valid || a->move_idx<0 || a->move_idx>=AI2G_MOVE_COUNT ||
        a->power_idx<0 || a->power_idx>=AI2G_POWER_COUNT ||
        a->first_hit_agent<-1 || a->first_hit_agent>1) return 0;
    return a->cast_idx>=0 && a->cast_idx<AI2G_ATTACK_MAX_CASTS &&
        a->cast_idx<ai2g_powers[a->power_idx].cast_count;
}

/* Power.do_power, preserving branch order, half-frame timing and first-change lookup. */
AI2G_HD static inline int ai2g_attack_step(AI2NEpisode *g,AI2GCombatPlayer e[2],int i,int *done) {
    if (!g || !e || !done || i<0 || i>1) return AI2G_PLAYER_ERR_INVALID;
    AI2GAttackRuntime *a=&e[i].attack;
    if (!ai2g_attack_indices_valid(a)) return AI2G_PLAYER_ERR_ATTACK_DATA;
    const AI2GPowerData *p=&ai2g_powers[a->power_idx];
    if (a->cast_idx<0 || a->cast_idx>=p->cast_count) return AI2G_PLAYER_ERR_ATTACK_DATA;
    const AI2GCastData *c=&ai2g_casts[p->first_cast+a->cast_idx];
    AI2CBody *body=&g->world.bodies[g->players[i].body];
    a->move_facing=g->players[i].ms.facing;
    e[i].gravity_scale=(p->flags&2) ? 0 : 1; /* The source body attribute is not consumed by integration. */
    e[i].attribute_presence|=AI2G_ATTR_GRAVITY_SCALE;
    *done=0;
    int next=-1,transitioning=0,rc=0;
    const int holding=g->players[i].input_state[ai2g_moves[a->move_idx].action_key];
    const AI2GCastData *last=&ai2g_casts[p->first_cast+p->cast_count-1];
    if ((p->flags&1) && ((!holding && a->total_frame_count>p->min_charge) || a->total_frame_count>last->startup)) {
        next=p->on_miss;
    } else {
        const int frame=a->cast_frame[a->cast_idx];
        const AI2GFrameChange *f=0;
        for (int k=0;k<c->change_count;++k) if (ai2g_changes[c->first_change+k].frame==frame) {
            f=&ai2g_changes[c->first_change+k]; break;
        }
        a->dealt_target_exists=f && f->dealt_active;
        a->dealt_target_components=3;
        a->dealt_target_x=a->dealt_target_exists ? f->dealt_x : 0;
        a->dealt_target_y=a->dealt_target_exists ? f->dealt_y : 0;
        a->dealt_target_mult=a->dealt_target_exists ? f->dealt_mult : 0;
        double hx=0,hy=0;
        if (!a->dealt_target_exists) {
            const double angle=ai2g_radians(c->hit_angle!=0 ? c->hit_angle : p->hit_angle);
            hx=cos(angle)*a->move_facing; hy=-sin(angle);
        }
        const int startup=frame<c->startup;
        const int active=!startup && (frame<c->startup+c->attack || c->must_be_held);
        const double damage=((p->flags&512) && c->attack!=0) ? c->damage/c->attack : c->damage;
        if (p->flags&128) for (int j=0;j<2;++j) if ((a->all_hit_mask&(1u<<j)) &&
                g->players[j].ms.state_id!=AI2_STATE_DODGE && !(a->power_hit_mask&(1u<<j))) {
            a->power_hit_mask|=1u<<j;
            const double force=ai2g_force(c,g->players[j].damage);
            if ((rc=ai2g_apply_damage(g,e,i,j,damage,p->stun_time,hx*force,hy*force))) return rc;
            e[j].gravity_scale=(p->flags&4) ? 0 : 1;
            e[j].attribute_presence|=AI2G_ATTR_GRAVITY_SCALE;
        }
        if (p->flags&64) ai2g_body_velocity(g,i,ai2_mov_move_toward(body->v.x,
            g->players[i].input_state[31]*g->mcfg.move_speed,g->mcfg.in_air_ease),body->v.y);
        ai2g_frame_changes(g,e,i,f,(p->flags&256)!=0);
        if (active) {
            if ((rc=ai2g_emit(e,i,AI2G_EVENT_ATTACK_ACTIVE,-1,-1,0))) return rc;
            g->players[i].ms.facing=a->move_facing;
            int collided=0;
            if (a->is_switching_casts) a->is_switching_casts=0;
            else for (int k=0;k<c->point_count;++k) {
                const AI2GPointData *pt=&ai2g_points[c->first_point+k];
                collided=ai2n_body_position(body).y+ai2g_units(pt->y)>1.54;
            }
            for (int k=0;k<c->point_count;++k) {
                const AI2GPointData *pt=&ai2g_points[c->first_point+k];
                a->past_point_x=ai2n_body_position(body).x+ai2g_units(pt->x)*a->move_facing;
                a->past_point_y=ai2n_body_position(body).y+ai2g_units(pt->y);
            }
            a->past_point_count=c->point_count;
            if (c->must_be_held && a->move_idx%11+2!=ai2g_get_move(g,i)) {
                transitioning=1;
                if (p->on_miss!=-1) return AI2G_PLAYER_ERR_SOURCE_EXCEPTION;
            }
            if (collided) { transitioning=1; next=p->on_ground!=-1 ? p->on_ground : p->on_miss; }
            int hitbox_hit=0;
            const int j=1-i;
            for (int k=0;k<c->hitbox_count;++k) {
                const AI2GHitboxData *hb=&ai2g_hitboxes[c->first_hitbox+k];
                const AI2GCapsule hit={ai2n_body_position(body).x+ai2g_units(hb->x)*a->move_facing,
                    ai2n_body_position(body).y+ai2g_units(hb->y),ai2g_units(hb->width),ai2g_units(hb->height)};
                const AI2GCapsule hurt={e[j].hurtbox_x,e[j].hurtbox_y,e[j].hurtbox_width,e[j].hurtbox_height};
                if (!ai2g_capsule_intersects(&hurt,&hit) || g->players[j].ms.state_id==AI2_STATE_DODGE) continue;
                hitbox_hit=1; e[i].attack_contacts++;
                if ((rc=ai2g_emit(e,i,AI2G_EVENT_ATTACK_CONTACT,j,k,0))) return rc;
                if (!a->hit_anyone && (p->flags&32)) {
                    const double angle=ai2g_radians(p->on_hit_angle);
                    ai2g_body_velocity(g,i,cos(angle)*p->on_hit_speed,sin(angle));
                }
                a->hit_anyone=1;
                const double force=ai2g_force(c,g->players[j].damage);
                if (p->flags&512) return AI2G_PLAYER_ERR_SOURCE_EXCEPTION;
                if (!(a->power_hit_mask&(1u<<j))) {
                    if (a->first_hit_agent<0) a->first_hit_agent=j;
                    if ((rc=ai2g_apply_damage(g,e,i,j,damage,p->stun_time,hx*force,hy*force))) return rc;
                    e[j].gravity_scale=(p->flags&4) ? 0 : 1;
                    e[j].attribute_presence|=AI2G_ATTR_GRAVITY_SCALE;
                    a->power_hit_mask|=1u<<j;
                }
                if (!(a->all_hit_mask&(1u<<j))) { e[j].just_got_hit=1; a->all_hit_mask|=1u<<j; }
            }
            if (hitbox_hit && (p->flags&16)) next=p->on_hit!=-1 ? p->on_hit : p->on_miss;
            if (a->cast_idx==p->cast_count-1 && p->on_hit==-1 && p->on_miss==-1) a->frames_into_recovery++;
        }
        a->cast_frame[a->cast_idx]++;
        a->in_recovery=!transitioning && !active && !startup;
        a->in_recovery_present=1;
        if (a->in_recovery) {
            if (a->cast_idx==p->cast_count-1) {
                const int recovery=p->fixed_recovery+(int)floor(p->recovery/1.426);
                if (a->frames_into_recovery>=recovery) {
                    if (p->on_hit==-1 && p->on_miss==-1) *done=1;
                    else if (a->hit_anyone) next=p->on_hit!=-1 ? p->on_hit : p->on_miss;
                    else next=p->on_miss;
                } else a->frames_into_recovery++;
            } else { a->cast_idx++; a->is_switching_casts=1; }
        }
    }
    a->total_frame_count++;
    if (next!=-1) {
        if ((rc=ai2g_power_begin(a,next))) return rc;
        if ((rc=ai2g_emit(e,i,AI2G_EVENT_POWER_CHANGED,-1,-1,0))) return rc;
    }
    a->move_frame++;
    return 0;
}

AI2G_HD static inline void ai2g_copy_air(AI2MovementState *s,int source,int target) {
    for (int field=AI2_MOV_SLOT_JUMP_TIMER;field<=AI2_MOV_SLOT_RECOVERIES_LEFT;++field)
        ai2_mov_slot_set(s,target,field,ai2_mov_slot_get(s,source,field,0));
}

/* Source base class: killbox precedes pending stun; pending stun REPLACES duration. */
AI2G_HD static inline int ai2g_interrupt(AI2NEpisode *g,AI2GCombatPlayer e[2],int i,int *target) {
    AI2MovementState *s=&g->players[i].ms;
    const int current=s->state_id;
    if (fabs(s->x)>floor(g->mcfg.stage_width_tiles/2.0) || fabs(s->y)>floor(g->mcfg.stage_height_tiles/2.0)) {
        *target=AI2_STATE_KO; return 1;
    }
    const int stun=ai2_mov_slot_get(s,current,AI2_MOV_SLOT_STUN_TIME_STORED,0);
    if (stun>0) {
        if (current==AI2G_STATE_STUN) e[i].hit_during_stun++;
        e[i].stun_frames=stun;
        e[i].attribute_presence|=AI2G_ATTR_STUN_FRAMES;
        ai2_mov_slot_set(s,current,AI2_MOV_SLOT_STUN_TIME_STORED,0);
        if (s->bank[current].present[AI2_MOV_SLOT_JUMPS_LEFT])
            ai2_mov_slot_set(s,AI2G_STATE_STUN,AI2_MOV_SLOT_JUMPS_LEFT,ai2_mov_slot_get(s,current,AI2_MOV_SLOT_JUMPS_LEFT,0));
        *target=AI2G_STATE_STUN; return 1;
    }
    *target=-1; return 0;
}

AI2G_HD static inline int ai2g_common(AI2NEpisode *g,AI2GCombatPlayer e[2],int i,int *target) {
    const int r=ai2g_interrupt(g,e,i,target);
    if (r) return r;
    AI2MovementState *s=&g->players[i].ms;
    for (int f=AI2_MOV_SLOT_INVINCIBLE_TIMER;f<=AI2_MOV_SLOT_DODGE_COOLDOWN;++f) {
        const int v=ai2_mov_slot_get(s,s->state_id,f,0);
        ai2_mov_slot_set(s,s->state_id,f,v>0 ? v-1 : 0);
    }
    return 0;
}

AI2G_HD static inline int ai2g_transition(AI2NEpisode *g,AI2GCombatPlayer e[2],int i,int target,AI2NRng *rng) {
    AI2MovementState *s=&g->players[i].ms;
    const int old=s->state_id;
    uint32_t events=0;
    int rc=0;
    ai2_mov_common_reset(s,target,old);
    ai2_mov_exit(s,old,&events);
    if (old==AI2G_STATE_ATTACK && (rc=ai2g_emit(e,i,AI2G_EVENT_ATTACK_ENDED,-1,-1,0))) return rc;
    s->state_id=target;
    if (target==AI2G_STATE_ATTACK) {
        ai2_mov_slot_set(s,target,AI2_MOV_SLOT_DASH_TIMER,g->mcfg.dash_time);
        if (!g->players[i].input_state[32]) s->facing=ai2_mov_sign_from_direction(g->players[i].input_state[31]);
        e[i].attack_seed=ai2n_rng_randint(rng,1,12);
        e[i].attribute_presence|=AI2G_ATTR_ATTACK_SEED;
    } else if (target==AI2G_STATE_STUN) {
        ai2_mov_slot_set(s,target,AI2_MOV_SLOT_IS_BASE,1);
        if ((rc=ai2g_emit(e,i,AI2G_EVENT_STUN,i,-1,e[i].stun_frames))) return rc;
    } else if (target==AI2G_STATE_TAUNT) {
        e[i].taunt_timer=30;
        e[i].taunt_seed=ai2n_rng_randint(rng,0,2);
        e[i].attribute_presence|=AI2G_ATTR_TAUNT_TIMER|AI2G_ATTR_TAUNT_SEED;
    } else if (target==AI2_STATE_KO) {
        ai2_mov_slot_set(s,target,AI2_MOV_SLOT_KO_TIMER,90);
        s->stocks--; /* Source does not clamp at zero. */
        s->velocity_func=AI2_VELOCITY_NO_GRAVITY; s->vx=0; s->vy=0;
        if ((rc=ai2g_emit(e,i,AI2G_EVENT_KO,i,-1,0))) return rc;
    } else ai2_mov_enter(s,&g->mcfg,target,&events);
    if (rng && rng->error) return AI2G_PLAYER_ERR_RNG;
    return 0;
}

AI2G_HD static inline int ai2g_start_attack(AI2NEpisode *g,AI2GCombatPlayer e[2],int i,int move,int aerial) {
    AI2MovementState *s=&g->players[i].ms;
    if (aerial) {
        ai2_mov_slot_set(s,AI2G_STATE_ATTACK,AI2_MOV_SLOT_JUMPS_LEFT,ai2_mov_slot_get(s,s->state_id,AI2_MOV_SLOT_JUMPS_LEFT,0));
        ai2_mov_slot_set(s,AI2G_STATE_ATTACK,AI2_MOV_SLOT_RECOVERIES_LEFT,ai2_mov_slot_get(s,s->state_id,AI2_MOV_SLOT_RECOVERIES_LEFT,0));
    } else {
        ai2_mov_slot_set(s,AI2G_STATE_ATTACK,AI2_MOV_SLOT_JUMP_TIMER,0);
        ai2_mov_slot_set(s,AI2G_STATE_ATTACK,AI2_MOV_SLOT_JUMPS_LEFT,2);
        ai2_mov_slot_set(s,AI2G_STATE_ATTACK,AI2_MOV_SLOT_RECOVERIES_LEFT,1);
    }
    return ai2g_attack_begin(g,e,i,move);
}

/* All source player states. Unchanged navigation branches reuse the proven reducer;
 * attack/taunt hooks are inserted at their original superclass priority points. */
AI2G_HD static inline int ai2g_state_physics(AI2NEpisode *g,AI2GCombatPlayer e[2],int i,
                                            const AI2MovementInput *in,int *target) {
    AI2MovementState *s=&g->players[i].ms;
    const AI2MovementConfig *c=&g->mcfg;
    const int current=s->state_id;
    uint32_t events=0;
    int rc=0;
    if (current==AI2_STATE_KO) {
        (void)ai2g_common(g,e,i,target); /* KO deliberately ignores the base return value. */
        const int timer=ai2_mov_slot_get(s,current,AI2_MOV_SLOT_KO_TIMER,0)-1;
        ai2_mov_slot_set(s,current,AI2_MOV_SLOT_KO_TIMER,timer);
        if (timer<=0) {
            ai2g_body_position(g,i,s->start_x,s->start_y); s->vx=0; s->vy=0;
            g->players[i].damage=0; g->players[i].weapon=0;
            e[i].damage_taken_this_stock=0; s->target_vx=0; s->target_vy=0;
            ai2_mov_slot_set(s,AI2_STATE_IN_AIR,AI2_MOV_SLOT_JUMPS_LEFT,0);
            ai2_mov_slot_set(s,AI2_STATE_IN_AIR,AI2_MOV_SLOT_RECOVERIES_LEFT,0);
            *target=AI2_STATE_IN_AIR;
            return ai2g_emit(e,i,AI2G_EVENT_RESPAWN,i,-1,0);
        }
        *target=-1; return 0;
    }
    if (ai2g_interrupt(g,e,i,target)) return 0;
    if (current==AI2G_STATE_ATTACK || current==AI2G_STATE_STUN || current==AI2G_STATE_TAUNT) {
        (void)ai2g_common(g,e,i,target);
        if (current==AI2G_STATE_ATTACK) {
            int done=0;
            if ((rc=ai2g_attack_step(g,e,i,&done))) return rc;
            if (done) {
                if (ai2n_is_on_floor(g,i)) *target=ai2_mov_ground_state_for_input(in);
                else {
                    *target=AI2_STATE_IN_AIR;
                    if (s->bank[current].present[AI2_MOV_SLOT_JUMPS_LEFT]) {
                        ai2g_copy_air(s,current,AI2_STATE_IN_AIR);
                        ai2_mov_slot_set(s,AI2_STATE_IN_AIR,AI2_MOV_SLOT_JUMP_TIMER,0);
                    }
                }
            }
        } else if (current==AI2G_STATE_STUN) {
            e[i].stun_frames=e[i].stun_frames>0 ? e[i].stun_frames-1 : 0;
            s->vx=ai2_mov_move_toward(s->vx,0,c->in_air_ease/1.5);
            if (e[i].stun_frames>0) {
                if (in->grounded && s->vy>0) s->vy=-s->vy*.7;
            } else if (in->grounded) *target=ai2_mov_ground_state_for_input(in);
            else {
                *target=AI2_STATE_IN_AIR;
                ai2_mov_slot_set(s,AI2_STATE_IN_AIR,AI2_MOV_SLOT_JUMPS_LEFT,
                    ai2n_imax(1,ai2_mov_slot_get(s,current,AI2_MOV_SLOT_JUMPS_LEFT,1)));
            }
        } else {
            e[i].taunt_timer=e[i].taunt_timer>0 ? e[i].taunt_timer-1 : 0;
            /* Python tests the bound is_grounded method, not its return value. */
            if (e[i].taunt_timer<=0) *target=ai2_mov_ground_state_for_input(in);
        }
        return 0;
    }
    if (current==AI2_STATE_IN_AIR || current==AI2_STATE_AIR_TURNAROUND) {
        (void)ai2g_common(g,e,i,target);
        const int direction=in->decoded[31];
        if (ai2_mov_slot_get(s,current,AI2_MOV_SLOT_IS_BASE,1) && ai2_mov_turn_check(s->facing,direction)) {
            ai2g_copy_air(s,current,AI2_STATE_AIR_TURNAROUND); *target=AI2_STATE_AIR_TURNAROUND; return 0;
        }
        s->vy=(in->decoded[30]<0 && s->vy>0) ? ai2_mov_move_toward(s->vy,c->max_fall_speed,c->fast_fall_ease) : ai2n_pymin(s->vy,c->max_fall_speed);
        s->vx=ai2_mov_move_toward(s->vx,direction*c->move_speed,c->in_air_ease);
        if (in->grounded) { *target=ai2_mov_ground_state_for_input(in); return 0; }
        int jump=ai2_mov_slot_get(s,current,AI2_MOV_SLOT_JUMP_TIMER,0);
        int left=ai2_mov_slot_get(s,current,AI2_MOV_SLOT_JUMPS_LEFT,0);
        if (in->decoded[14] && jump<=0 && left>0) {
            s->vy=-c->jump_speed; s->facing=ai2_mov_sign_from_direction(direction); jump=c->jump_cooldown; left--;
        }
        jump=jump>0 ? jump-1 : 0;
        ai2_mov_slot_set(s,current,AI2_MOV_SLOT_JUMP_TIMER,jump); ai2_mov_slot_set(s,current,AI2_MOV_SLOT_JUMPS_LEFT,left);
        if (in->decoded[16] && ai2_mov_slot_get(s,current,AI2_MOV_SLOT_DODGE_COOLDOWN,0)<=0) {
            ai2_mov_slot_set(s,current,AI2_MOV_SLOT_DODGE_COOLDOWN,c->air_dodge_cooldown);
            ai2g_copy_air(s,current,AI2_STATE_DODGE);
            ai2_mov_slot_set(s,AI2_STATE_DODGE,AI2_MOV_SLOT_DODGE_IS_GROUNDED,0);
            *target=AI2_STATE_DODGE; return 0;
        }
        const int move=ai2g_get_move(g,i);
        if (move!=1) {
            int recovery=ai2_mov_slot_get(s,current,AI2_MOV_SLOT_RECOVERIES_LEFT,0);
            if (move!=11 || recovery>0) {
                if (move==11) ai2_mov_slot_set(s,current,AI2_MOV_SLOT_RECOVERIES_LEFT,recovery-1);
                if ((rc=ai2g_start_attack(g,e,i,move,1))) return rc;
                *target=AI2G_STATE_ATTACK; return 0;
            }
        }
        if (current==AI2_STATE_AIR_TURNAROUND) {
            const int timer=ai2_mov_slot_get(s,current,AI2_MOV_SLOT_TURNAROUND_TIMER,0);
            if (timer<=0) {
                const int last=ai2_mov_hstate_last_int(in->decoded[33]),old=s->facing;
                s->facing=-s->facing;
                *target=(-old==last) ? AI2_STATE_IN_AIR : (old==last ? AI2_STATE_AIR_TURNAROUND : AI2_STATE_IN_AIR);
                ai2g_copy_air(s,current,*target);
            } else ai2_mov_slot_set(s,current,AI2_MOV_SLOT_TURNAROUND_TIMER,timer-1);
        }
        return 0;
    }
    const int ground_control=current==AI2_STATE_WALKING || current==AI2_STATE_STANDING ||
        current==AI2_STATE_TURNAROUND || current==AI2_STATE_SPRINTING || current==AI2_STATE_DASH;
    if (ground_control && (in->decoded[7] || in->decoded[8] || in->decoded[19])) {
        rc=ai2_mov_ground_common(c,s,in,current,target);
        if (rc<0) return AI2G_PLAYER_ERR_INVALID;
        if (rc) return 0;
        const int move=ai2g_get_move(g,i);
        if (move!=1) {
            if ((rc=ai2g_start_attack(g,e,i,move,0))) return rc;
            *target=AI2G_STATE_ATTACK;
        } else *target=AI2G_STATE_TAUNT;
        return 0;
    }
    rc=ai2_mov_state_physics(c,s,in,current,target,&events);
    return rc<0 ? AI2G_PLAYER_ERR_INVALID : 0;
}

/* Floor queries refresh shape caches, so all handles they may touch (including
 * a damage target's handles) must be checked before invoking either query. */
AI2G_HD static inline int ai2g_player_handles_valid(const AI2NEpisode *g) {
    const AI2CWorld *w=&g->world;
    if (w->body_count<1 || w->body_count>AI2C_MAX_BODIES ||
        w->shape_count<2 || w->shape_count>AI2C_MAX_SHAPES ||
        g->platform_body<0 || g->platform_body>=w->body_count) return 0;
    for (int j=0;j<2;++j)
        if (g->players[j].body<0 || g->players[j].body>=w->body_count) return 0;
    const int shapes[5]={0,1,g->platform_shape,g->players[0].shape,g->players[1].shape};
    for (int j=0;j<5;++j) {
        if (shapes[j]<0 || shapes[j]>=w->shape_count) return 0;
        const AI2CShape *shape=&w->shapes[shapes[j]];
        if (shape->body<0 || shape->body>=w->body_count ||
            shape->count<1 || shape->count>AI2C_MAX_VERTS) return 0;
    }
    return 1;
}

AI2G_HD static inline int ai2g_player_physics(AI2NEpisode *g,AI2GCombatPlayer e[2],int i,AI2NRng *rng) {
    if (!g || !e || !rng || i<0 || i>1) return AI2G_PLAYER_ERR_INVALID;
    if (!ai2g_player_handles_valid(g)) return AI2G_PLAYER_ERR_INVALID;
    if ((e[i].attack.valid || g->players[i].ms.state_id==AI2G_STATE_ATTACK) &&
        !ai2g_attack_indices_valid(&e[i].attack)) return AI2G_PLAYER_ERR_ATTACK_DATA;
    AI2NPlayer *p=&g->players[i]; AI2MovementState *s=&p->ms;
    AI2CBody *b=&g->world.bodies[p->body];
    const AI2CBody *platform=&g->world.bodies[g->platform_body];
    AI2MovementInput in;
    for (int k=0;k<AI2_INPUT_WIDTH;++k) in.decoded[k]=p->input_state[k];
    in.grounded=(uint8_t)ai2n_is_on_floor(g,i); in.platform_overlap=(uint8_t)ai2n_platform_overlap(g,i);
    in.platform_vx=platform->v.x; in.platform_vy=platform->v.y;
    s->x=ai2n_body_position(b).x; s->y=ai2n_body_position(b).y; s->vx=b->v.x; s->vy=b->v.y;
    s->velocity_func=b->velocity_mode;
    if (s->state_id<0 || s->state_id>=AI2_MOVEMENT_STATE_COUNT || !ai2_mov_finite_state(&g->mcfg,s,&in)) return AI2G_PLAYER_ERR_INVALID;
    int target=-1,rc=ai2g_state_physics(g,e,i,&in,&target);
    if (rc) return rc;
    e[i].hurtbox_x=ai2n_body_position(b).x; e[i].hurtbox_y=ai2n_body_position(b).y;
    s->vx=(s->vx+s->damage_vx)+s->target_vx; s->vy=(s->vy+s->damage_vy)+s->target_vy;
    if (target>=0 && (rc=ai2g_transition(g,e,i,target,rng))) return rc;
    s->prev_x=ai2n_body_position(b).x; s->prev_y=ai2n_body_position(b).y;
    s->damage_vx=0; s->damage_vy=0; s->target_vx=0; s->target_vy=0;
    b->v=ai2c_v(s->vx,s->vy); b->velocity_mode=s->velocity_func;
    return 0;
}

#endif
