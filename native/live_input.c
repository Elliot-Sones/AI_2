#define PY_SSIZE_T_CLEAN
#include <Python.h>

enum { KEY_NAMES, PREV_STATE, KEY_STATUS, HELD, JUST_PRESSED, JUST_RELEASED,
       RAW_VERTICAL, RAW_HORIZONTAL, NO_HORIZONTAL, HORIZONTAL_STATE,
       REGISTER_KEYS, KEY_W, KEY_A, KEY_S, KEY_D, NAME_COUNT };
static const char *attribute_names[NAME_COUNT] = {
    "key_names", "prev_state", "key_status", "held", "just_pressed", "just_released",
    "raw_vertical", "raw_horizontal", "no_horizontal", "horizontal_state",
    "register_keys", "W", "A", "S", "D"
};
typedef struct { PyObject *names[NAME_COUNT]; PyObject *half; } InputState;

/* Live Python objects remain authoritative. Preserve Python/NumPy boolean
 * result types and the frozen enum's register_keys ordering. No raw pointers,
 * alternate random stream, action mask or gameplay timing changes. */
static int held(PyObject *statuses, PyObject *key, PyObject *held_name) {
    PyObject *status = PyDict_GetItemWithError(statuses, key);
    PyObject *value;
    int result;
    if (!status) { PyErr_SetObject(PyExc_KeyError, key); return -1; }
    value = PyObject_GetAttr(status, held_name);
    if (!value) return -1;
    result = PyObject_IsTrue(value);
    Py_DECREF(value);
    return result;
}

static int set_float(PyObject *object, PyObject *name, double value) {
    PyObject *number = PyFloat_FromDouble(value);
    int result;
    if (!number) return -1;
    result = PyObject_SetAttr(object, name, number);
    Py_DECREF(number);
    return result;
}

static PyObject *update(PyObject *module, PyObject *args) {
    PyObject *handler, *action, *names = NULL, *keys = NULL;
    PyObject *previous = NULL, *statuses = NULL;
    PyObject *horizontal = NULL, *next_horizontal = NULL;
    InputState *state = PyModule_GetState(module);
    PyObject **attrs = state->names;
    int w, a, s, d;
    if (!PyArg_ParseTuple(args, "OO:update", &handler, &action)) return NULL;
    names = PyObject_GetAttr(handler, attrs[KEY_NAMES]);
    if (!names) goto fail;
    keys = PySequence_Fast(names, "key_names must be a sequence");
    if (!keys) goto fail;
    previous = PyObject_GetAttr(handler, attrs[PREV_STATE]);
    statuses = PyObject_GetAttr(handler, attrs[KEY_STATUS]);
    if (!previous || !statuses) goto fail;
    if (!PyDict_Check(previous) || !PyDict_Check(statuses)) {
        PyErr_SetString(PyExc_TypeError, "input state must contain dictionaries");
        goto fail;
    }
    for (Py_ssize_t i = 0; i < PySequence_Fast_GET_SIZE(keys); ++i) {
        PyObject *key = PySequence_Fast_GET_ITEM(keys, i);
        PyObject *index = PyLong_FromSsize_t(i);
        PyObject *value, *current, *old, *status, *pressed, *released;
        int old_truth, current_truth, error;
        if (!index) goto fail;
        value = PyObject_GetItem(action, index);
        Py_DECREF(index);
        if (!value) goto fail;
        current = PyObject_RichCompare(value, state->half, Py_GT);
        Py_DECREF(value);
        if (!current) goto fail;
        old = PyDict_GetItemWithError(previous, key);
        status = PyDict_GetItemWithError(statuses, key);
        if (!old || !status) {
            if (!PyErr_Occurred()) PyErr_SetObject(PyExc_KeyError, key);
            Py_DECREF(current);
            goto fail;
        }
        old_truth = PyObject_IsTrue(old);
        current_truth = PyObject_IsTrue(current);
        if (old_truth < 0 || current_truth < 0) { Py_DECREF(current); goto fail; }
        pressed = old_truth ? Py_False : current;
        released = old_truth ? (current_truth ? Py_False : Py_True) : old;
        error = PyObject_SetAttr(status, attrs[JUST_PRESSED], pressed);
        if (!error) error = PyObject_SetAttr(status, attrs[JUST_RELEASED], released);
        if (!error) error = PyObject_SetAttr(status, attrs[HELD], current);
        if (!error) error = PyDict_SetItem(previous, key, current);
        Py_DECREF(current);
        if (error) goto fail;
    }
    w = held(statuses, attrs[KEY_W], attrs[HELD]); a = held(statuses, attrs[KEY_A], attrs[HELD]);
    s = held(statuses, attrs[KEY_S], attrs[HELD]); d = held(statuses, attrs[KEY_D], attrs[HELD]);
    if (w < 0 || a < 0 || s < 0 || d < 0) goto fail;
    if (set_float(handler, attrs[RAW_VERTICAL], (double)(w - s)) < 0 ||
        set_float(handler, attrs[RAW_HORIZONTAL], (double)(d - a)) < 0 ||
        PyObject_SetAttr(handler, attrs[NO_HORIZONTAL], (!d && !a) ? Py_True : Py_False) < 0) goto fail;
    horizontal = PyObject_GetAttr(handler, attrs[HORIZONTAL_STATE]);
    if (!horizontal) goto fail;
    next_horizontal = PyObject_CallMethodObjArgs(horizontal, attrs[REGISTER_KEYS],
        PyDict_GetItemWithError(statuses, attrs[KEY_A]), PyDict_GetItemWithError(statuses, attrs[KEY_D]), NULL);
    if (!next_horizontal || PyObject_SetAttr(handler, attrs[HORIZONTAL_STATE], next_horizontal) < 0) goto fail;
    Py_DECREF(names); Py_DECREF(keys); Py_DECREF(previous); Py_DECREF(statuses);
    Py_DECREF(horizontal); Py_DECREF(next_horizontal);
    Py_RETURN_NONE;
fail:
    Py_XDECREF(names); Py_XDECREF(keys); Py_XDECREF(previous); Py_XDECREF(statuses);
    Py_XDECREF(horizontal); Py_XDECREF(next_horizontal);
    return NULL;
}

static PyMethodDef methods[] = {
    {"update", update, METH_VARARGS, "Update live frozen PlayerInputHandler objects."},
    {NULL, NULL, 0, NULL}
};
static int traverse(PyObject *module, visitproc visit, void *arg) {
    InputState *state = PyModule_GetState(module);
    for (int i = 0; i < NAME_COUNT; ++i) Py_VISIT(state->names[i]);
    Py_VISIT(state->half);
    return 0;
}
static int clear(PyObject *module) {
    InputState *state = PyModule_GetState(module);
    for (int i = 0; i < NAME_COUNT; ++i) Py_CLEAR(state->names[i]);
    Py_CLEAR(state->half);
    return 0;
}
static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "_ai2_live_input", NULL, sizeof(InputState), methods,
    NULL, traverse, clear, NULL
};
PyMODINIT_FUNC PyInit__ai2_live_input(void) {
    PyObject *result = PyModule_Create(&module);
    InputState *state;
    if (!result) return NULL;
    state = PyModule_GetState(result);
    for (int i = 0; i < NAME_COUNT; ++i) {
        state->names[i] = PyUnicode_InternFromString(attribute_names[i]);
        if (!state->names[i]) { Py_DECREF(result); return NULL; }
    }
    state->half = PyFloat_FromDouble(0.5);
    if (!state->half) { Py_DECREF(result); return NULL; }
    return result;
}
