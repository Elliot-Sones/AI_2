#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdint.h>

typedef struct {
    double x;
    double y;
} AI2CpVect;

typedef AI2CpVect (*AI2CpBodyVecFn)(void *body);

enum {
    NAME_ACTIVE_WEAPON,
    NAME_BODY,
    NAME_DAMAGE,
    NAME_DODGE_TIMER,
    NAME_ENV,
    NAME_FACING,
    NAME_GET,
    NAME_GET_OBS,
    NAME_IS_ON_FLOOR,
    NAME_JUMPS_LEFT,
    NAME_MOVE_TYPE,
    NAME_PLATFORM,
    NAME_POSITION,
    NAME_RECOVERIES_LEFT,
    NAME_SPAWNERS,
    NAME_STATE,
    NAME_STATE_MAPPING,
    NAME_STOCKS,
    NAME_STUN_FRAMES,
    NAME_VELOCITY,
    NAME_WEAPON,
    NAME_WEAPON_CONTROLLER,
    NAME_WEAPON_MAPPING,
    NAME_WEAPON_NAME,
    NAME_WORLD_POS,
    NAME_X,
    NAME_Y,
    NAME___NAME__,
    NAME_COUNT
};

static const char *attribute_names[NAME_COUNT] = {
    "active_weapon",
    "body",
    "damage",
    "dodge_timer",
    "env",
    "facing",
    "get",
    "get_obs",
    "is_on_floor",
    "jumps_left",
    "move_type",
    "PLATFORM",
    "position",
    "recoveries_left",
    "spawners",
    "state",
    "state_mapping",
    "stocks",
    "stun_frames",
    "velocity",
    "weapon",
    "weapon_controller",
    "weapon_mapping",
    "weapon_name",
    "world_pos",
    "x",
    "y",
    "__name__",
};

typedef struct {
    PyObject *names[NAME_COUNT];
} LiveObsState;

static double clamp_double(double value, double lo, double hi) {
    if (value < lo) return lo;
    if (value > hi) return hi;
    return value;
}

typedef struct {
    PyObject *list;
    Py_ssize_t index;
} ObsBuilder;

static int append_double(ObsBuilder *builder, double value) {
    PyObject *item = PyFloat_FromDouble(value);
    if (item == NULL) return -1;
    if (builder->index >= 32) {
        Py_DECREF(item);
        PyErr_SetString(PyExc_RuntimeError, "native get_obs wrote more than 32 fields");
        return -1;
    }
    PyList_SET_ITEM(builder->list, builder->index, item);
    builder->index += 1;
    return 0;
}

static PyObject *required_attr(PyObject *obj, PyObject *name) {
    return PyObject_GetAttr(obj, name);
}

static PyObject *optional_attr(PyObject *obj, PyObject *name) {
    PyObject *value = PyObject_GetAttr(obj, name);
    if (value == NULL && PyErr_ExceptionMatches(PyExc_AttributeError)) {
        PyErr_Clear();
        return NULL;
    }
    return value;
}

static int py_number_as_double(PyObject *obj, double *out) {
    double value = PyFloat_AsDouble(obj);
    if (PyErr_Occurred()) return -1;
    *out = value;
    return 0;
}

static int attr_as_double(PyObject *obj, PyObject *name, double *out) {
    PyObject *value = required_attr(obj, name);
    int status;
    if (value == NULL) return -1;
    status = py_number_as_double(value, out);
    Py_DECREF(value);
    return status;
}

static int optional_attr_as_double(PyObject *obj, PyObject *name, double fallback, double *out) {
    PyObject *value = optional_attr(obj, name);
    int status;
    if (value == NULL) {
        if (PyErr_Occurred()) return -1;
        *out = fallback;
        return 0;
    }
    status = py_number_as_double(value, out);
    Py_DECREF(value);
    return status;
}

static int body_vec2_attr(PyObject *body, PyObject *vec_name, LiveObsState *state, double *x, double *y) {
    PyObject *vec = required_attr(body, vec_name);
    int status = -1;
    if (vec == NULL) return -1;
    if (attr_as_double(vec, state->names[NAME_X], x) == 0 &&
        attr_as_double(vec, state->names[NAME_Y], y) == 0) {
        status = 0;
    }
    Py_DECREF(vec);
    return status;
}

static int append_body_pair(ObsBuilder *builder, PyObject *body, PyObject *vec_name,
                            LiveObsState *state, double x_lo, double x_hi,
                            double y_lo, double y_hi) {
    double x, y;
    if (body_vec2_attr(body, vec_name, state, &x, &y) != 0) return -1;
    return append_double(builder, clamp_double(x, x_lo, x_hi)) ||
           append_double(builder, clamp_double(y, y_lo, y_hi));
}

static int append_direct_pair(ObsBuilder *builder, AI2CpBodyVecFn getter, void *body,
                              double x_lo, double x_hi, double y_lo, double y_hi) {
    AI2CpVect value = getter(body);
    return append_double(builder, clamp_double(value.x, x_lo, x_hi)) ||
           append_double(builder, clamp_double(value.y, y_lo, y_hi));
}

static int append_optional_state_attr(ObsBuilder *builder, PyObject *state_obj, PyObject *name) {
    double value;
    if (optional_attr_as_double(state_obj, name, 0.0, &value) != 0) return -1;
    return append_double(builder, value);
}

static int append_spawner_slot(ObsBuilder *builder, PyObject *spawner,
                               PyObject *dropped_type, LiveObsState *state) {
    PyObject *active_weapon = required_attr(spawner, state->names[NAME_ACTIVE_WEAPON]);
    double x_norm = 0.0, y_norm = 0.0, weapon_type = 0.0;
    int is_dropped;
    if (active_weapon == NULL) return -1;
    if (active_weapon != Py_None) {
        PyObject *world_pos = required_attr(spawner, state->names[NAME_WORLD_POS]);
        PyObject *x_obj, *y_obj;
        double x, y;
        Py_DECREF(active_weapon);
        if (world_pos == NULL) return -1;
        x_obj = PySequence_GetItem(world_pos, 0);
        y_obj = PySequence_GetItem(world_pos, 1);
        Py_DECREF(world_pos);
        if (x_obj == NULL || y_obj == NULL) {
            Py_XDECREF(x_obj);
            Py_XDECREF(y_obj);
            return -1;
        }
        if (py_number_as_double(x_obj, &x) != 0 || py_number_as_double(y_obj, &y) != 0) {
            Py_DECREF(x_obj);
            Py_DECREF(y_obj);
            return -1;
        }
        Py_DECREF(x_obj);
        Py_DECREF(y_obj);
        x_norm = clamp_double(x, -18.0, 18.0);
        y_norm = clamp_double(y, -7.0, 7.0);
        weapon_type = 1.0;
        is_dropped = PyObject_IsInstance(spawner, dropped_type);
        if (is_dropped < 0) return -1;
        if (is_dropped) {
            PyObject *weapon_name = required_attr(spawner, state->names[NAME_WEAPON_NAME]);
            if (weapon_name == NULL) return -1;
            if (PyUnicode_Check(weapon_name)) {
                int spear = PyUnicode_CompareWithASCIIString(weapon_name, "Spear");
                if (spear == -1 && PyErr_Occurred()) {
                    Py_DECREF(weapon_name);
                    return -1;
                }
                if (spear == 0) {
                    weapon_type = 2.0;
                } else {
                    int hammer = PyUnicode_CompareWithASCIIString(weapon_name, "Hammer");
                    if (hammer == -1 && PyErr_Occurred()) {
                        Py_DECREF(weapon_name);
                        return -1;
                    }
                    if (hammer == 0) weapon_type = 3.0;
                }
            }
            Py_DECREF(weapon_name);
        }
    } else {
        Py_DECREF(active_weapon);
    }
    return append_double(builder, x_norm) ||
           append_double(builder, y_norm) ||
           append_double(builder, weapon_type);
}

static int append_spawners(ObsBuilder *builder, PyObject *player,
                           PyObject *dropped_type, LiveObsState *state) {
    PyObject *env = required_attr(player, state->names[NAME_ENV]);
    PyObject *controller, *spawners;
    int status = 0;
    if (env == NULL) return -1;
    controller = required_attr(env, state->names[NAME_WEAPON_CONTROLLER]);
    Py_DECREF(env);
    if (controller == NULL) return -1;
    spawners = required_attr(controller, state->names[NAME_SPAWNERS]);
    Py_DECREF(controller);
    if (spawners == NULL) return -1;
    for (int i = 0; i < 4; i += 1) {
        PyObject *spawner = PySequence_GetItem(spawners, i);
        if (spawner == NULL) {
            if (PyErr_ExceptionMatches(PyExc_IndexError)) {
                PyErr_Clear();
                status = append_double(builder, 0.0) ||
                         append_double(builder, 0.0) ||
                         append_double(builder, 0.0);
            } else {
                status = -1;
            }
        } else {
            status = append_spawner_slot(builder, spawner, dropped_type, state);
            Py_DECREF(spawner);
        }
        if (status != 0) break;
    }
    Py_DECREF(spawners);
    return status;
}

static PyObject *native_get_obs_impl(PyObject *player, PyObject *dropped_type,
                                     PyObject *facing_right, LiveObsState *state,
                                     void *player_body_ptr, void *platform_body_ptr,
                                     AI2CpBodyVecFn body_position,
                                     AI2CpBodyVecFn body_velocity) {
    PyObject *body = NULL, *facing = NULL, *is_floor_method = NULL, *grounded_obj = NULL;
    PyObject *state_obj = NULL, *state_mapping = NULL, *state_type_name = NULL;
    PyObject *state_index_obj = NULL, *zero = NULL, *damage_obj = NULL, *stocks_obj = NULL;
    PyObject *weapon_mapping = NULL, *weapon = NULL, *weapon_index_obj = NULL;
    PyObject *platform = NULL, *platform_body = NULL;
    ObsBuilder builder;
    double value;
    int equal, grounded;

    if (!PyType_Check(dropped_type)) {
        PyErr_SetString(PyExc_TypeError, "dropped_spawner_type must be a type");
        return NULL;
    }

    builder.list = PyList_New(32);
    builder.index = 0;
    if (builder.list == NULL) return NULL;

    if (player_body_ptr != NULL && body_position != NULL && body_velocity != NULL) {
        if (append_direct_pair(&builder, body_position, player_body_ptr, -18.0, 18.0, -7.0, 7.0) != 0 ||
            append_direct_pair(&builder, body_velocity, player_body_ptr, -10.0, 10.0, -10.0, 10.0) != 0) goto fail;
    } else {
        body = required_attr(player, state->names[NAME_BODY]);
        if (body == NULL) goto fail;
        if (body == Py_None) {
            PyErr_SetString(PyExc_ValueError, "player body is required");
            goto fail;
        }
        if (append_body_pair(&builder, body, state->names[NAME_POSITION], state, -18.0, 18.0, -7.0, 7.0) != 0 ||
            append_body_pair(&builder, body, state->names[NAME_VELOCITY], state, -10.0, 10.0, -10.0, 10.0) != 0) goto fail;
        Py_CLEAR(body);
    }

    facing = required_attr(player, state->names[NAME_FACING]);
    if (facing == NULL) goto fail;
    equal = PyObject_RichCompareBool(facing, facing_right, Py_EQ);
    Py_CLEAR(facing);
    if (equal < 0 || append_double(&builder, equal ? 1.0 : 0.0) != 0) goto fail;

    is_floor_method = required_attr(player, state->names[NAME_IS_ON_FLOOR]);
    if (is_floor_method == NULL) goto fail;
    grounded_obj = PyObject_CallNoArgs(is_floor_method);
    Py_CLEAR(is_floor_method);
    if (grounded_obj == NULL) goto fail;
    grounded = PyObject_IsTrue(grounded_obj);
    Py_CLEAR(grounded_obj);
    if (grounded < 0 ||
        append_double(&builder, grounded ? 1.0 : 0.0) != 0 ||
        append_double(&builder, grounded ? 0.0 : 1.0) != 0) goto fail;

    state_obj = required_attr(player, state->names[NAME_STATE]);
    if (state_obj == NULL) goto fail;
    if (append_optional_state_attr(&builder, state_obj, state->names[NAME_JUMPS_LEFT]) != 0) goto fail;

    state_mapping = required_attr(player, state->names[NAME_STATE_MAPPING]);
    if (state_mapping == NULL) goto fail;
    state_type_name = PyObject_GetAttr((PyObject *)Py_TYPE(state_obj), state->names[NAME___NAME__]);
    zero = PyLong_FromLong(0);
    if (state_type_name == NULL || zero == NULL) goto fail;
    state_index_obj = PyObject_CallMethodObjArgs(state_mapping, state->names[NAME_GET], state_type_name, zero, NULL);
    Py_CLEAR(state_type_name);
    Py_CLEAR(zero);
    Py_CLEAR(state_mapping);
    if (state_index_obj == NULL || py_number_as_double(state_index_obj, &value) != 0) goto fail;
    Py_CLEAR(state_index_obj);
    if (append_double(&builder, value) != 0 ||
        append_optional_state_attr(&builder, state_obj, state->names[NAME_RECOVERIES_LEFT]) != 0 ||
        append_optional_state_attr(&builder, state_obj, state->names[NAME_DODGE_TIMER]) != 0 ||
        append_optional_state_attr(&builder, state_obj, state->names[NAME_STUN_FRAMES]) != 0) goto fail;

    damage_obj = required_attr(player, state->names[NAME_DAMAGE]);
    if (damage_obj == NULL || py_number_as_double(damage_obj, &value) != 0) goto fail;
    Py_CLEAR(damage_obj);
    if (append_double(&builder, value / 700.0) != 0) goto fail;

    stocks_obj = required_attr(player, state->names[NAME_STOCKS]);
    if (stocks_obj == NULL || py_number_as_double(stocks_obj, &value) != 0) goto fail;
    Py_CLEAR(stocks_obj);
    if (append_double(&builder, value) != 0 ||
        append_optional_state_attr(&builder, state_obj, state->names[NAME_MOVE_TYPE]) != 0) goto fail;
    Py_CLEAR(state_obj);

    weapon_mapping = required_attr(player, state->names[NAME_WEAPON_MAPPING]);
    weapon = required_attr(player, state->names[NAME_WEAPON]);
    if (weapon_mapping == NULL || weapon == NULL) goto fail;
    weapon_index_obj = PyObject_GetItem(weapon_mapping, weapon);
    Py_CLEAR(weapon_mapping);
    Py_CLEAR(weapon);
    if (weapon_index_obj == NULL || py_number_as_double(weapon_index_obj, &value) != 0) goto fail;
    Py_CLEAR(weapon_index_obj);
    if (append_double(&builder, value) != 0 ||
        append_spawners(&builder, player, dropped_type, state) != 0) goto fail;

    if (platform_body_ptr != NULL && body_position != NULL && body_velocity != NULL) {
        if (append_direct_pair(&builder, body_position, platform_body_ptr, -18.0, 18.0, -7.0, 7.0) != 0 ||
            append_direct_pair(&builder, body_velocity, platform_body_ptr, -10.0, 10.0, -10.0, 10.0) != 0) goto fail;
    } else {
        platform = required_attr(player, state->names[NAME_PLATFORM]);
        if (platform == NULL) goto fail;
        platform_body = required_attr(platform, state->names[NAME_BODY]);
        Py_CLEAR(platform);
        if (platform_body == NULL) goto fail;
        if (platform_body == Py_None) {
            PyErr_SetString(PyExc_ValueError, "platform body is required");
            goto fail;
        }
        if (append_body_pair(&builder, platform_body, state->names[NAME_POSITION], state, -18.0, 18.0, -7.0, 7.0) != 0 ||
            append_body_pair(&builder, platform_body, state->names[NAME_VELOCITY], state, -10.0, 10.0, -10.0, 10.0) != 0) goto fail;
        Py_CLEAR(platform_body);
    }

    if (builder.index != 32) {
        PyErr_SetString(PyExc_RuntimeError, "native get_obs did not write exactly 32 fields");
        goto fail;
    }
    return builder.list;

fail:
    Py_XDECREF(body);
    Py_XDECREF(facing);
    Py_XDECREF(is_floor_method);
    Py_XDECREF(grounded_obj);
    Py_XDECREF(state_obj);
    Py_XDECREF(state_mapping);
    Py_XDECREF(state_type_name);
    Py_XDECREF(state_index_obj);
    Py_XDECREF(zero);
    Py_XDECREF(damage_obj);
    Py_XDECREF(stocks_obj);
    Py_XDECREF(weapon_mapping);
    Py_XDECREF(weapon);
    Py_XDECREF(weapon_index_obj);
    Py_XDECREF(platform);
    Py_XDECREF(platform_body);
    Py_DECREF(builder.list);
    return NULL;
}

static PyObject *native_get_obs(PyObject *module, PyObject *args) {
    PyObject *player, *dropped_type, *facing_right;
    LiveObsState *state = PyModule_GetState(module);
    if (!PyArg_ParseTuple(args, "OOO:get_obs", &player, &dropped_type, &facing_right)) return NULL;
    return native_get_obs_impl(player, dropped_type, facing_right, state, NULL, NULL, NULL, NULL);
}

typedef struct {
    PyObject_HEAD
    PyObject *module;
    PyObject *player;
    PyObject *player_body;
    PyObject *platform_body;
    PyObject *dropped_type;
    PyObject *facing_right;
    void *player_body_ptr;
    void *platform_body_ptr;
    AI2CpBodyVecFn body_position;
    AI2CpBodyVecFn body_velocity;
} PreparedGetObs;

static void PreparedGetObs_dealloc(PreparedGetObs *self) {
    PyObject_GC_UnTrack(self);
    Py_CLEAR(self->module);
    Py_CLEAR(self->player);
    Py_CLEAR(self->player_body);
    Py_CLEAR(self->platform_body);
    Py_CLEAR(self->dropped_type);
    Py_CLEAR(self->facing_right);
    self->player_body_ptr = NULL;
    self->platform_body_ptr = NULL;
    self->body_position = NULL;
    self->body_velocity = NULL;
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static int PreparedGetObs_traverse(PreparedGetObs *self, visitproc visit, void *arg) {
    Py_VISIT(self->module);
    Py_VISIT(self->player);
    Py_VISIT(self->player_body);
    Py_VISIT(self->platform_body);
    Py_VISIT(self->dropped_type);
    Py_VISIT(self->facing_right);
    return 0;
}

static int PreparedGetObs_clear(PreparedGetObs *self) {
    Py_CLEAR(self->module);
    Py_CLEAR(self->player);
    Py_CLEAR(self->player_body);
    Py_CLEAR(self->platform_body);
    Py_CLEAR(self->dropped_type);
    Py_CLEAR(self->facing_right);
    self->player_body_ptr = NULL;
    self->platform_body_ptr = NULL;
    self->body_position = NULL;
    self->body_velocity = NULL;
    return 0;
}

static PyObject *PreparedGetObs_call(PyObject *callable, PyObject *args, PyObject *kwargs) {
    PreparedGetObs *self = (PreparedGetObs *)callable;
    LiveObsState *state;
    (void)kwargs;
    if (PyTuple_GET_SIZE(args) != 0) {
        PyErr_SetString(PyExc_TypeError, "prepared get_obs takes no arguments");
        return NULL;
    }
    if (self->module == NULL || self->player == NULL || self->dropped_type == NULL ||
        self->facing_right == NULL || self->player_body_ptr == NULL ||
        self->platform_body_ptr == NULL || self->body_position == NULL ||
        self->body_velocity == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "prepared get_obs has been cleared");
        return NULL;
    }
    state = PyModule_GetState(self->module);
    return native_get_obs_impl(self->player, self->dropped_type, self->facing_right, state,
                               self->player_body_ptr, self->platform_body_ptr,
                               self->body_position, self->body_velocity);
}

static PyTypeObject PreparedGetObsType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "_ai2_native_live_observation.PreparedGetObs",
    .tp_basicsize = sizeof(PreparedGetObs),
    .tp_itemsize = 0,
    .tp_dealloc = (destructor)PreparedGetObs_dealloc,
    .tp_call = PreparedGetObs_call,
    .tp_traverse = (traverseproc)PreparedGetObs_traverse,
    .tp_clear = (inquiry)PreparedGetObs_clear,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC,
    .tp_doc = "Prepared native Player.get_obs callable.",
    .tp_free = PyObject_GC_Del,
};

static PyObject *native_prepare(PyObject *module, PyObject *args) {
    PyObject *player, *dropped_type, *facing_right, *player_body, *platform_body;
    unsigned long long player_body_addr, platform_body_addr, position_addr, velocity_addr;
    PreparedGetObs *prepared;
    if (!PyArg_ParseTuple(args, "OOOOOKKKK:prepare", &player, &dropped_type, &facing_right,
                          &player_body, &platform_body, &player_body_addr,
                          &platform_body_addr, &position_addr, &velocity_addr)) return NULL;
    if (!PyType_Check(dropped_type)) {
        PyErr_SetString(PyExc_TypeError, "dropped_spawner_type must be a type");
        return NULL;
    }
    if (player_body_addr == 0 || platform_body_addr == 0 || position_addr == 0 || velocity_addr == 0) {
        PyErr_SetString(PyExc_ValueError, "prepared get_obs requires non-null body/function pointers");
        return NULL;
    }
    prepared = PyObject_GC_New(PreparedGetObs, &PreparedGetObsType);
    if (prepared == NULL) return NULL;
    prepared->module = NULL;
    prepared->player = NULL;
    prepared->player_body = NULL;
    prepared->platform_body = NULL;
    prepared->dropped_type = NULL;
    prepared->facing_right = NULL;
    prepared->player_body_ptr = NULL;
    prepared->platform_body_ptr = NULL;
    prepared->body_position = NULL;
    prepared->body_velocity = NULL;
    Py_INCREF(module);
    Py_INCREF(player);
    Py_INCREF(player_body);
    Py_INCREF(platform_body);
    Py_INCREF(dropped_type);
    Py_INCREF(facing_right);
    prepared->module = module;
    prepared->player = player;
    prepared->player_body = player_body;
    prepared->platform_body = platform_body;
    prepared->dropped_type = dropped_type;
    prepared->facing_right = facing_right;
    prepared->player_body_ptr = (void *)(uintptr_t)player_body_addr;
    prepared->platform_body_ptr = (void *)(uintptr_t)platform_body_addr;
    prepared->body_position = (AI2CpBodyVecFn)(uintptr_t)position_addr;
    prepared->body_velocity = (AI2CpBodyVecFn)(uintptr_t)velocity_addr;
    PyObject_GC_Track(prepared);
    return (PyObject *)prepared;
}

static PyMethodDef LiveObservationMethods[] = {
    {"get_obs", native_get_obs, METH_VARARGS, "Return frozen Player.get_obs as 32 Python floats."},
    {"prepare", native_prepare, METH_VARARGS, "Prepare a no-argument native Player.get_obs callable."},
    {NULL, NULL, 0, NULL}
};

static int liveobs_traverse(PyObject *module, visitproc visit, void *arg) {
    LiveObsState *state = PyModule_GetState(module);
    (void)arg;
    for (int i = 0; i < NAME_COUNT; i += 1) Py_VISIT(state->names[i]);
    return 0;
}

static int liveobs_clear(PyObject *module) {
    LiveObsState *state = PyModule_GetState(module);
    for (int i = 0; i < NAME_COUNT; i += 1) Py_CLEAR(state->names[i]);
    return 0;
}

static struct PyModuleDef LiveObservationModule = {
    PyModuleDef_HEAD_INIT,
    "_ai2_native_live_observation",
    "Native live Player.get_obs helper.",
    sizeof(LiveObsState),
    LiveObservationMethods,
    NULL,
    liveobs_traverse,
    liveobs_clear,
    NULL
};

PyMODINIT_FUNC PyInit__ai2_native_live_observation(void) {
    PyObject *module;
    LiveObsState *state;
    if (PyType_Ready(&PreparedGetObsType) < 0) return NULL;
    module = PyModule_Create(&LiveObservationModule);
    if (module == NULL) return NULL;
    state = PyModule_GetState(module);
    for (int i = 0; i < NAME_COUNT; i += 1) {
        state->names[i] = PyUnicode_InternFromString(attribute_names[i]);
        if (state->names[i] == NULL) {
            Py_DECREF(module);
            return NULL;
        }
    }
    Py_INCREF(&PreparedGetObsType);
    if (PyModule_AddObject(module, "PreparedGetObs", (PyObject *)&PreparedGetObsType) < 0) {
        Py_DECREF(&PreparedGetObsType);
        Py_DECREF(module);
        return NULL;
    }
    return module;
}
