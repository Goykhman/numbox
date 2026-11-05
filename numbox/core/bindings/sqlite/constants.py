"""SQLite's integer constants under their C names, for Python code and jitted code alike.

The module holds the result codes, the column type codes, the flags of ``sqlite3_open_v2``, ``sqlite3_blob_open`` and
``sqlite3_trace_v2``, the destructor values ``SQLITE_STATIC`` and ``SQLITE_TRANSIENT``, the flags that register a SQL
function, and the constraint operators of a virtual table's ``xBestIndex``. Import a constant by name, as in
``from numbox.core.bindings.sqlite.constants import SQLITE_OK``. The names are the C names, in capitals, so they do
not collide with the bindings, which use the lowercase names of SQLite's functions. numba reads a module-level
integer as a constant, so jitted code can use these names directly. SQLite has not changed these values between
3.34.0, the oldest version in numbox's test matrix, and the current release.
"""

__all__ = [
    "SQLITE_OK", "SQLITE_ERROR", "SQLITE_INTERNAL", "SQLITE_PERM", "SQLITE_ABORT",
    "SQLITE_BUSY", "SQLITE_LOCKED", "SQLITE_NOMEM", "SQLITE_READONLY",
    "SQLITE_INTERRUPT", "SQLITE_IOERR", "SQLITE_CORRUPT", "SQLITE_NOTFOUND",
    "SQLITE_FULL", "SQLITE_CANTOPEN", "SQLITE_PROTOCOL", "SQLITE_EMPTY",
    "SQLITE_SCHEMA", "SQLITE_TOOBIG", "SQLITE_CONSTRAINT", "SQLITE_MISMATCH",
    "SQLITE_MISUSE", "SQLITE_NOLFS", "SQLITE_AUTH", "SQLITE_FORMAT",
    "SQLITE_RANGE", "SQLITE_NOTADB", "SQLITE_NOTICE", "SQLITE_WARNING",
    "SQLITE_ROW", "SQLITE_DONE",
    "SQLITE_INTEGER", "SQLITE_FLOAT", "SQLITE_TEXT", "SQLITE_BLOB", "SQLITE_NULL",
    "SQLITE_OPEN_READONLY", "SQLITE_OPEN_READWRITE", "SQLITE_OPEN_CREATE",
    "SQLITE_OPEN_URI", "SQLITE_OPEN_MEMORY", "SQLITE_OPEN_NOMUTEX",
    "SQLITE_OPEN_FULLMUTEX", "SQLITE_OPEN_SHAREDCACHE", "SQLITE_OPEN_PRIVATECACHE",
    "SQLITE_BLOB_READONLY", "SQLITE_BLOB_READWRITE",
    "SQLITE_TRACE_STMT", "SQLITE_TRACE_PROFILE", "SQLITE_TRACE_ROW",
    "SQLITE_TRACE_CLOSE",
    "SQLITE_STATIC", "SQLITE_TRANSIENT",
    "SQLITE_UTF8", "SQLITE_DETERMINISTIC", "SQLITE_DIRECTONLY", "SQLITE_INNOCUOUS",
    "SQLITE_SUBTYPE", "SQLITE_RESULT_SUBTYPE",
    "SQLITE_INDEX_CONSTRAINT_EQ", "SQLITE_INDEX_CONSTRAINT_GT",
    "SQLITE_INDEX_CONSTRAINT_LE", "SQLITE_INDEX_CONSTRAINT_LT",
    "SQLITE_INDEX_CONSTRAINT_GE", "SQLITE_INDEX_CONSTRAINT_NE",
    "SQLITE_INDEX_CONSTRAINT_ISNULL", "SQLITE_INDEX_CONSTRAINT_IS",
]

# === Primary result codes (sqlite3.h) ===
SQLITE_OK = 0
SQLITE_ERROR = 1
SQLITE_INTERNAL = 2
SQLITE_PERM = 3
SQLITE_ABORT = 4
SQLITE_BUSY = 5
SQLITE_LOCKED = 6
SQLITE_NOMEM = 7
SQLITE_READONLY = 8
SQLITE_INTERRUPT = 9
SQLITE_IOERR = 10
SQLITE_CORRUPT = 11
SQLITE_NOTFOUND = 12
SQLITE_FULL = 13
SQLITE_CANTOPEN = 14
SQLITE_PROTOCOL = 15
SQLITE_EMPTY = 16
SQLITE_SCHEMA = 17
SQLITE_TOOBIG = 18
SQLITE_CONSTRAINT = 19
SQLITE_MISMATCH = 20
SQLITE_MISUSE = 21
SQLITE_NOLFS = 22
SQLITE_AUTH = 23
SQLITE_FORMAT = 24
SQLITE_RANGE = 25
SQLITE_NOTADB = 26
SQLITE_NOTICE = 27
SQLITE_WARNING = 28
SQLITE_ROW = 100
SQLITE_DONE = 101

# === Column type codes (sqlite3_column_type return values) ===
SQLITE_INTEGER = 1
SQLITE_FLOAT = 2
SQLITE_TEXT = 3
SQLITE_BLOB = 4
SQLITE_NULL = 5

# === sqlite3_open_v2 flags (combinable with bitwise OR) ===
SQLITE_OPEN_READONLY = 0x00000001
SQLITE_OPEN_READWRITE = 0x00000002
SQLITE_OPEN_CREATE = 0x00000004
SQLITE_OPEN_URI = 0x00000040
SQLITE_OPEN_MEMORY = 0x00000080
SQLITE_OPEN_NOMUTEX = 0x00008000
SQLITE_OPEN_FULLMUTEX = 0x00010000
SQLITE_OPEN_SHAREDCACHE = 0x00020000
SQLITE_OPEN_PRIVATECACHE = 0x00040000

# === sqlite3_blob_open flags (the values of its flags argument) ===
SQLITE_BLOB_READONLY = 0
SQLITE_BLOB_READWRITE = 1

# === sqlite3_trace_v2 event mask bits ===
SQLITE_TRACE_STMT = 0x01
SQLITE_TRACE_PROFILE = 0x02
SQLITE_TRACE_ROW = 0x04
SQLITE_TRACE_CLOSE = 0x08

# === Destructor values for the text and BLOB bindings of bind.py and result.py ===
SQLITE_STATIC = 0
SQLITE_TRANSIENT = -1

# === Flags of sqlite3_create_function_v2 and sqlite3_create_window_function ===
SQLITE_UTF8 = 1
SQLITE_DETERMINISTIC = 0x800
SQLITE_DIRECTONLY = 0x80000
SQLITE_INNOCUOUS = 0x200000
# SQLITE_SUBTYPE (SQLite 3.30.0 and later) registers a function that reads its arguments' subtypes with
# sqlite3_value_subtype. If the flag is missing, that call may return 0 for an argument that has a subtype.
# SQLITE_RESULT_SUBTYPE (3.45.0 and later) registers a function that sets its result's subtype with
# sqlite3_result_subtype. A SQLite built with -DSQLITE_STRICT_SUBTYPE=1, such as the conda-forge build, makes the
# function fail at that call when this flag is missing. The strict build checks only SQLITE_RESULT_SUBTYPE, in
# sqlite3_result_subtype (SQLite's src/vdbeapi.c); sqlite3_value_subtype checks no flag. Older versions ignore both.
SQLITE_SUBTYPE = 0x100000
SQLITE_RESULT_SUBTYPE = 0x1000000

# === Constraint operators of sqlite3_index_info, which xBestIndex reads to pass constraints to xFilter ===
SQLITE_INDEX_CONSTRAINT_EQ = 2
SQLITE_INDEX_CONSTRAINT_GT = 4
SQLITE_INDEX_CONSTRAINT_LE = 8
SQLITE_INDEX_CONSTRAINT_LT = 16
SQLITE_INDEX_CONSTRAINT_GE = 32
SQLITE_INDEX_CONSTRAINT_NE = 68
SQLITE_INDEX_CONSTRAINT_ISNULL = 71
SQLITE_INDEX_CONSTRAINT_IS = 72
