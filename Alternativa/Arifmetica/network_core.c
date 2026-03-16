#include <Python.h>

#define MAX_QUEUE 100000
#define MAX_STEPS 100000

static PyObject* run_network_fast(PyObject* self, PyObject* args)
{
    PyObject *ntype;
    PyObject *state;
    PyObject *param;
    PyObject *out_index;
    PyObject *conn_to;
    PyObject *inputs;

    if(!PyArg_ParseTuple(args,"OOOOOO",
        &ntype,&state,&param,&out_index,&conn_to,&inputs))
        return NULL;

    long n = PyList_Size(state);

    long queue[MAX_QUEUE];
    long qsize = 0;

    long steps = 0;
    long queue_max = 0;

    /* подаём вход */

    long inputs_n = PyList_Size(inputs);

    for(long i=0;i<inputs_n;i++)
    {
        PyObject* pair = PyList_GetItem(inputs,i);

        long neuron =
            PyLong_AsLong(PyTuple_GetItem(pair,0));

        long value =
            PyLong_AsLong(PyTuple_GetItem(pair,1));

        if(neuron < n)
        {
            PyList_SetItem(
                state,
                neuron,
                PyLong_FromLong(value)
            );

            queue[qsize++] = neuron;
        }
    }

    queue_max = qsize;

    /* симуляция */

    while(qsize > 0 && steps < MAX_STEPS)
    {
        long neuron = queue[--qsize];

        long s = PyLong_AsLong(
            PyList_GetItem(state,neuron));

        long t = PyLong_AsLong(
            PyList_GetItem(ntype,neuron));

        long p = PyLong_AsLong(
            PyList_GetItem(param,neuron));

        long new_state;

        if(t == 0)
            new_state = s ^ (p & 3);
        else if(t == 1)
            new_state = (s + p) & 3;
        else
            new_state = (s * (p | 1)) & 3;

        if(new_state == s)
            goto next;

        PyList_SetItem(
            state,
            neuron,
            PyLong_FromLong(new_state)
        );

        PyObject* edges =
            PyList_GetItem(out_index,neuron);

        long ecount = PyList_Size(edges);

        for(long i=0;i<ecount;i++)
        {
            long edge =
                PyLong_AsLong(PyList_GetItem(edges,i));

            long dst =
                PyLong_AsLong(PyList_GetItem(conn_to,edge));

            long dst_state =
                PyLong_AsLong(PyList_GetItem(state,dst));

            dst_state ^= new_state;

            PyList_SetItem(
                state,
                dst,
                PyLong_FromLong(dst_state)
            );

            if(qsize < MAX_QUEUE)
                queue[qsize++] = dst;
        }

        next:

        steps++;

        if(qsize > queue_max)
            queue_max = qsize;

        if(queue_max > MAX_QUEUE)
            Py_RETURN_NONE;
    }

    if(steps >= MAX_STEPS)
        Py_RETURN_NONE;

    long out =
        PyLong_AsLong(
            PyList_GetItem(state,n-1)) & 1;

    return Py_BuildValue("iii",out,steps,queue_max);
}











static PyObject* process(
    PyObject* self,
    PyObject* args
){
    PyObject *ntype;
    PyObject *state;
    PyObject *param;
    PyObject *out_index;
    PyObject *conn_to;
    PyObject *queue;
    int i;

if(!PyArg_ParseTuple(
    args,
    "iOOOOOO",
    &i,
    &ntype,
    &state,
    &param,
    &out_index,
    &conn_to,
    &queue
)) return NULL;

    long s = PyLong_AsLong(PyList_GetItem(state,i));
    long p = PyLong_AsLong(PyList_GetItem(param,i));
    long t = PyLong_AsLong(PyList_GetItem(ntype,i));

    long new_state;

    if(t==0)
        new_state = s ^ (p & 3);
    else if(t==1)
        new_state = (s + p) & 3;
    else
        new_state = (s * (p | 1)) & 3;

    if(new_state == s)
        Py_RETURN_NONE;

    PyList_SetItem(state,i,PyLong_FromLong(new_state));

    PyObject *edges = PyList_GetItem(out_index,i);
    Py_ssize_t n = PyList_Size(edges);

    for(Py_ssize_t k=0;k<n;k++)
    {
        long edge =
            PyLong_AsLong(PyList_GetItem(edges,k));

        long dst =
            PyLong_AsLong(PyList_GetItem(conn_to,edge));

        long v =
            PyLong_AsLong(PyList_GetItem(state,dst));

        PyList_SetItem(
            state,
            dst,
            PyLong_FromLong(v ^ new_state)
        );

        PyList_Append(queue,PyLong_FromLong(dst));
    }

    Py_RETURN_NONE;
}



static PyMethodDef methods[] = {
    {"run_network_fast", run_network_fast, METH_VARARGS, ""},
    {"process", process, METH_VARARGS, ""},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "network_core",
    NULL,
    -1,
    methods
};

PyMODINIT_FUNC PyInit_network_core(void)
{
    return PyModule_Create(&module);
}