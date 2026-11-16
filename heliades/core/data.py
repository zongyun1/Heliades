# SPDX-License-Identifier: LGPL-2.1-or-later
# Ported from Helios core (Context_data.cpp, Context.h data templates; Copyright (C) 2016-2026 Brian Bailey).
"""Typed primitive data and object data (mixed into Scene).

Storage matches the layout W1-B's copy/delete code already uses: ``Scene.primitive_data`` and
``Scene.object_data`` map ``label -> {id: TypedData}``. Helios keeps one type per label in a
Context-wide registry (``primitive_data_type_registry``) that is released once no primitive holds the
label any more; here that is the type of any entry of the label's dict, and an empty dict means the
lock is released. Every stored value is a list (Helios stores scalars as one-element vectors), with
elements of the C++ type: numpy int32 / uint32 / float32 / float64 scalars, float32 (vec) or int32
(int) arrays of length 2-4, ``str`` or ``bool``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..migration_registry import ports

f32 = np.float32

# helios::HeliosDataType
(TYPE_INT, TYPE_UINT, TYPE_FLOAT, TYPE_DOUBLE, TYPE_VEC2, TYPE_VEC3, TYPE_VEC4, TYPE_INT2, TYPE_INT3, TYPE_INT4,
 TYPE_STRING, TYPE_BOOL, TYPE_UNKNOWN) = range(13)

TYPE_BY_NAME = {"int": TYPE_INT, "uint": TYPE_UINT, "float": TYPE_FLOAT, "double": TYPE_DOUBLE, "vec2": TYPE_VEC2,
                "vec3": TYPE_VEC3, "vec4": TYPE_VEC4, "int2": TYPE_INT2, "int3": TYPE_INT3, "int4": TYPE_INT4,
                "string": TYPE_STRING, "bool": TYPE_BOOL}
TYPE_NAMES = {v: k for k, v in TYPE_BY_NAME.items()}
_NUMERIC = (TYPE_INT, TYPE_UINT, TYPE_FLOAT, TYPE_DOUBLE)
_VEC = {TYPE_VEC2: 2, TYPE_VEC3: 3, TYPE_VEC4: 4}
_IVEC = {TYPE_INT2: 2, TYPE_INT3: 3, TYPE_INT4: 4}
_COMPARATORS = ("==", ">", "<", ">=", "<=")


@dataclass
class TypedData:
    type: int
    values: list

    def front(self):
        return self.values[0]


def _type_code(dtype) -> int:
    if isinstance(dtype, str):
        return TYPE_BY_NAME[dtype]
    return int(dtype)


def infer_type(value) -> int:
    """C++ overload a Python value stands for: np.float32 / Python float -> float, np.float64 -> double,
    np.uint32 -> uint, other ints -> int, str -> string, bool -> bool, length-2..4 sequences -> vecN
    (float) or intN (all integer). Pass ``dtype`` explicitly where this is ambiguous."""
    if isinstance(value, (bool, np.bool_)):
        return TYPE_BOOL
    if isinstance(value, str):
        return TYPE_STRING
    if isinstance(value, np.float64):
        return TYPE_DOUBLE
    if isinstance(value, (float, np.floating)):
        return TYPE_FLOAT
    if isinstance(value, np.unsignedinteger):
        return TYPE_UINT
    if isinstance(value, (int, np.integer)):
        return TYPE_INT
    a = np.asarray(value)
    if a.ndim == 1 and 2 <= len(a) <= 4:
        if np.issubdtype(a.dtype, np.integer):
            return (TYPE_INT2, TYPE_INT3, TYPE_INT4)[len(a) - 2]
        return (TYPE_VEC2, TYPE_VEC3, TYPE_VEC4)[len(a) - 2]
    raise TypeError(f"setPrimitiveData/setObjectData: unsupported value {value!r}")


def _wrap_int(v, bits=32, signed=True):
    v = int(v) & ((1 << bits) - 1)
    if signed and v >= 1 << (bits - 1):
        v -= 1 << bits
    return v


def coerce(value, t: int):
    """The value as the C++ type ``t`` (the input already has that type, e.g. a Python float for float)."""
    if t == TYPE_INT:
        return np.int32(_wrap_int(value))
    if t == TYPE_UINT:
        return np.uint32(_wrap_int(value, signed=False))
    if t == TYPE_FLOAT:
        return f32(value)
    if t == TYPE_DOUBLE:
        return np.float64(value)
    if t in _VEC:
        a = np.asarray(value, f32).reshape(-1)
        if len(a) != _VEC[t]:
            raise ValueError(f"expected {_VEC[t]} components for {TYPE_NAMES[t]}")
        return a.copy()
    if t in _IVEC:
        a = np.asarray(value).reshape(-1)
        if len(a) != _IVEC[t]:
            raise ValueError(f"expected {_IVEC[t]} components for {TYPE_NAMES[t]}")
        return np.array([_wrap_int(x) for x in a], np.int32)
    if t == TYPE_STRING:
        return str(value)
    if t == TYPE_BOOL:
        return bool(value)
    raise TypeError(f"unknown data type {t}")


def static_cast(value, target: int):
    """storeDataWithTypeCasting: static_cast between the numeric scalar types."""
    if target in (TYPE_INT, TYPE_UINT):
        if isinstance(value, (np.floating, float)):
            value = int(value)                           # truncation toward zero (C++ float -> integer)
        return coerce(int(value), target)
    if target == TYPE_FLOAT:
        return f32(value)
    if target == TYPE_DOUBLE:
        return np.float64(value)
    return value


def _ids(x) -> list[int]:
    return [int(x)] if np.isscalar(x) else [int(i) for i in x]


def _compare(data, value, comparator: str) -> bool:
    if comparator == "==":
        return data == value
    if comparator == ">":
        return data > value
    if comparator == "<":
        return data < value
    if comparator == ">=":
        return data >= value
    return data <= value


class _TypedStore:
    """Shared logic of primitive data and object data; ``kind`` is 'Primitive' or 'Object'."""

    def __init__(self, scene, kind: str):
        self.scene, self.kind = scene, kind

    @property
    def store(self) -> dict:
        return self.scene.primitive_data if self.kind == "Primitive" else self.scene.object_data

    def exists_id(self, i: int) -> bool:
        return i in (self.scene.primitives if self.kind == "Primitive" else self.scene.objects)

    def registered_type(self, label: str):
        d = self.store.get(label)
        if not d:
            return None
        return next(iter(d.values())).type

    def register_or_validate(self, label: str, t: int) -> int:
        """registerOrValidate{Primitive,Object}DataType: the label's type, or ``t`` on first use; a
        different numeric type is cast to the registered one, anything else is an error."""
        expected = self.registered_type(label)
        if expected is None or expected == t:
            return t
        if not (t in _NUMERIC and expected in _NUMERIC):
            raise TypeError(f"ERROR (Context::set{self.kind}Data): {self.kind} data '{label}' already exists with type "
                            f"'{TYPE_NAMES[expected]}', so it cannot be assigned a value with type '{TYPE_NAMES[t]}'.")
        return expected

    def check_ids(self, ids):
        for i in ids:
            if not self.exists_id(i):
                what = "UUID" if self.kind == "Primitive" else "objID"
                raise ValueError(f"ERROR (Context::set{self.kind}Data): {what} of {i} does not exist in the Context. "
                                 f"No {self.kind.lower()} data was set.")

    def set(self, ids, label: str, value, dtype=None, vector=False):
        ids = _ids(ids)
        self.check_ids(ids)
        if vector:
            vals = list(value)
            t = _type_code(dtype) if dtype is not None else (infer_type(vals[0]) if vals else TYPE_FLOAT)
            vals = [coerce(v, t) for v in vals]
        else:
            t = _type_code(dtype) if dtype is not None else infer_type(value)
            vals = [coerce(value, t)]
        target = self.register_or_validate(label, t)
        if target != t:
            vals = [static_cast(v, target) for v in vals]
        d = self.store.setdefault(label, {})
        for i in ids:
            d[i] = TypedData(target, [v.copy() if isinstance(v, np.ndarray) else v for v in vals])

    def entry(self, i: int, label: str, who: str) -> TypedData:
        if not self.exists_id(i):
            raise ValueError(f"ERROR (Context::{who}): ID of {i} does not exist in the Context.")
        e = self.store.get(label, {}).get(i)
        if e is None:
            raise KeyError(f"ERROR (Context::{who}): {self.kind} data {label} does not exist for {self.kind.lower()} {i}.")
        return e

    def get(self, i: int, label: str, dtype=None, vector=False):
        e = self.entry(int(i), label, f"get{self.kind}Data")
        if dtype is not None and _type_code(dtype) != e.type:
            raise TypeError(f"ERROR (Context::get{self.kind}Data): Attempted to get data for type {TYPE_NAMES[_type_code(dtype)]}, "
                            f"but data {label} for {self.kind.lower()} {i} does not have type {TYPE_NAMES[e.type]}.")
        if vector:
            return [v.copy() if isinstance(v, np.ndarray) else v for v in e.values]
        v = e.front()
        return v.copy() if isinstance(v, np.ndarray) else v

    def exists(self, i: int, label: str) -> bool:
        i = int(i)
        if not self.exists_id(i):
            raise ValueError(f"ERROR (Context::does{self.kind}DataExist): ID of {i} does not exist in the Context.")
        return i in self.store.get(label, {})

    def clear(self, ids, label: str):
        for i in _ids(ids):
            if not self.exists_id(i):
                raise ValueError(f"ERROR (Context::clear{self.kind}Data): ID of {i} does not exist in the Context.")
            d = self.store.get(label)
            if d is not None:
                d.pop(i, None)
                if not d:
                    del self.store[label]

    def list_labels(self, i: int) -> list[str]:
        i = int(i)
        if not self.exists_id(i):
            raise ValueError(f"ERROR (Context::list{self.kind}Data): ID of {i} does not exist in the Context.")
        return sorted(lbl for lbl, d in self.store.items() if i in d)    # std::map<std::string, ...> order

    def filter(self, ids, label: str, value, comparator: str = "==", dtype=None) -> list[int]:
        """filter{Primitives,Objects}ByData: Helios walks the list from the back and swap-and-pops every
        rejected ID, so the output is a permutation of the kept IDs, reproduced here."""
        t = _type_code(dtype) if dtype is not None else infer_type(value)
        if t not in (TYPE_FLOAT, TYPE_DOUBLE, TYPE_INT, TYPE_UINT, TYPE_STRING):
            raise TypeError(f"filter{self.kind}sByData: unsupported filter value type {TYPE_NAMES.get(t, t)}")
        if t != TYPE_STRING and comparator not in _COMPARATORS:
            raise ValueError(f"ERROR (Context::filter{self.kind}sByData): Invalid comparator. Must be one of '==', '>', '<', '>=', or '<='.")
        fv = coerce(value, t)
        out = list(_ids(ids))
        reg = self.registered_type(label)
        d = self.store.get(label, {})
        for p in range(len(out) - 1, -1, -1):
            i = out[p]
            if not self.exists_id(i):                    # primitives.at(UUID) throws
                raise ValueError(f"ERROR (Context::does{self.kind}DataExist): ID of {i} does not exist in the Context.")
            if i in d and reg == t:
                data = d[i].front()
                if t == TYPE_STRING:
                    if data == fv:
                        continue
                elif _compare(data, fv, comparator):
                    continue
            out[p], out[-1] = out[-1], out[p]
            out.pop()
        return out


class DataMixin:
    """Context primitive-data and object-data API."""

    def _pdata(self) -> _TypedStore:
        return _TypedStore(self, "Primitive")

    def _odata(self) -> _TypedStore:
        return _TypedStore(self, "Object")

    # ---- primitive data ----------------------------------------------------------------------

    @ports("Context::setPrimitiveData(uint,const char*,const T&)", "Context::setPrimitiveData(const std::vector<uint>&,const char*,const T&)",
           "Context::registerOrValidatePrimitiveDataType", "Context::storeDataWithTypeCasting")
    def set_primitive_data(self, uuids, label: str, value, dtype=None):
        """Scalar data on one UUID or a list of UUIDs (all checked to exist before any is set).
        ``dtype`` ('int', 'uint', 'float', 'double', 'vec3', 'string', 'bool', ...) overrides the
        inferred C++ type (see ``infer_type``)."""
        self._pdata().set(uuids, label, value, dtype)

    @ports("Context::setPrimitiveData(uint,const char*,const std::vector<T>&)")
    def set_primitive_data_vector(self, uuids, label: str, values, dtype=None):
        self._pdata().set(uuids, label, values, dtype, vector=True)

    @ports("Context::getPrimitiveData(uint,const char*,T&)")
    def get_primitive_data(self, uuid: int, label: str, dtype=None):
        """The (first) value. With ``dtype``, raises unless the stored type is that type, as Helios does."""
        return self._pdata().get(uuid, label, dtype)

    @ports("Context::getPrimitiveData(uint,const char*,std::vector<T>&)")
    def get_primitive_data_vector(self, uuid: int, label: str, dtype=None) -> list:
        return self._pdata().get(uuid, label, dtype, vector=True)

    @ports("Context::doesPrimitiveDataExist")
    def does_primitive_data_exist(self, uuid: int, label: str) -> bool:
        return self._pdata().exists(uuid, label)

    @ports("Context::getPrimitiveDataType(uint,const char*)")
    def get_primitive_data_type(self, uuid: int, label: str) -> int:
        return self._pdata().entry(int(uuid), label, "getPrimitiveDataType").type

    @ports("Context::getPrimitiveDataType(const char*)")
    def get_primitive_data_type_registered(self, label: str) -> int:
        t = self._pdata().registered_type(label)
        if t is None:
            raise KeyError(f"ERROR (Context::getPrimitiveDataType): Primitive data {label} does not exist.")
        return t

    @ports("Context::getPrimitiveDataSize")
    def get_primitive_data_size(self, uuid: int, label: str) -> int:
        return len(self._pdata().entry(int(uuid), label, "getPrimitiveDataSize").values)

    @ports("Context::clearPrimitiveData(uint,const char*)", "Context::clearPrimitiveData(const std::vector<uint>&,const char*)")
    def clear_primitive_data(self, uuids, label: str):
        self._pdata().clear(uuids, label)

    @ports("Context::listPrimitiveData")
    def list_primitive_data(self, uuid: int) -> list[str]:
        return self._pdata().list_labels(uuid)

    @ports("Context::listAllPrimitiveDataLabels")
    def list_all_primitive_data_labels(self) -> list[str]:
        """Sorted (Helios iterates an unordered_map)."""
        return sorted(lbl for lbl, d in self.primitive_data.items() if d)

    @ports("Context::filterPrimitivesByData(const std::vector<uint>&,const std::string&,float,const std::string&)",
           "Context::filterPrimitivesByData(const std::vector<uint>&,const std::string&,double,const std::string&)",
           "Context::filterPrimitivesByData(const std::vector<uint>&,const std::string&,int,const std::string&)",
           "Context::filterPrimitivesByData(const std::vector<uint>&,const std::string&,uint,const std::string&)",
           "Context::filterPrimitivesByData(const std::vector<uint>&,const std::string&,const std::string&)")
    def filter_primitives_by_data(self, uuids, label: str, value, comparator: str = "==", dtype=None) -> list[int]:
        """Keep the UUIDs whose ``label`` data has the filter value's type and passes the comparison
        (strings: equality). The overload is chosen by ``dtype`` or by ``infer_type(value)``."""
        return self._pdata().filter(uuids, label, value, comparator, dtype)

    # ---- object data -------------------------------------------------------------------------

    @ports("Context::setObjectData(uint,const char*,const T&)", "Context::setObjectData(const std::vector<uint>&,const char*,const T&)",
           "Context::registerOrValidateObjectDataType", "Context::storeObjectDataWithTypeCasting")
    def set_object_data(self, oids, label: str, value, dtype=None):
        self._odata().set(oids, label, value, dtype)

    @ports("Context::setObjectData(uint,const char*,const std::vector<T>&)")
    def set_object_data_vector(self, oids, label: str, values, dtype=None):
        self._odata().set(oids, label, values, dtype, vector=True)

    @ports("Context::getObjectData(uint,const char*,T&)")
    def get_object_data(self, oid: int, label: str, dtype=None):
        return self._odata().get(oid, label, dtype)

    @ports("Context::getObjectData(uint,const char*,std::vector<T>&)")
    def get_object_data_vector(self, oid: int, label: str, dtype=None) -> list:
        return self._odata().get(oid, label, dtype, vector=True)

    @ports("Context::doesObjectDataExist")
    def does_object_data_exist(self, oid: int, label: str) -> bool:
        return self._odata().exists(oid, label)

    @ports("Context::getObjectDataType(uint,const char*)")
    def get_object_data_type(self, oid: int, label: str) -> int:
        return self._odata().entry(int(oid), label, "getObjectDataType").type

    @ports("Context::getObjectDataType(const char*)")
    def get_object_data_type_registered(self, label: str) -> int:
        t = self._odata().registered_type(label)
        if t is None:
            raise KeyError(f"ERROR (Context::getObjectDataType): Object data {label} does not exist.")
        return t

    @ports("Context::getObjectDataSize")
    def get_object_data_size(self, oid: int, label: str) -> int:
        return len(self._odata().entry(int(oid), label, "getObjectDataSize").values)

    @ports("Context::clearObjectData(uint,const char*)", "Context::clearObjectData(const std::vector<uint>&,const char*)")
    def clear_object_data(self, oids, label: str):
        self._odata().clear(oids, label)

    @ports("Context::listObjectData")
    def list_object_data(self, oid: int) -> list[str]:
        return self._odata().list_labels(oid)

    @ports("Context::listAllObjectDataLabels")
    def list_all_object_data_labels(self) -> list[str]:
        return sorted(lbl for lbl, d in self.object_data.items() if d)

    @ports("Context::copyObjectData")
    def copy_object_data(self, src: int, dst: int):
        """Replace all of dst's object data by copies of src's."""
        import copy as _copy
        for d in self.object_data.values():
            d.pop(dst, None)
            if src in d:
                d[dst] = _copy.deepcopy(d[src])

    @ports("Context::filterObjectsByData(const std::vector<uint>&,const std::string&,float,const std::string&)",
           "Context::filterObjectsByData(const std::vector<uint>&,const std::string&,double,const std::string&)",
           "Context::filterObjectsByData(const std::vector<uint>&,const std::string&,int,const std::string&)",
           "Context::filterObjectsByData(const std::vector<uint>&,const std::string&,uint,const std::string&)",
           "Context::filterObjectsByData(const std::vector<uint>&,const std::string&,const std::string&)")
    def filter_objects_by_data(self, oids, label: str, value, comparator: str = "==", dtype=None) -> list[int]:
        return self._odata().filter(oids, label, value, comparator, dtype)
