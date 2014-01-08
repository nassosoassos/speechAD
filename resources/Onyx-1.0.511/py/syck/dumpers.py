"""
syck.dumpers is a high-level wrapper for the Syck YAML emitter.
Do not use it directly, use the module 'syck' instead.
"""

import _syck

try:
    import cStringIO as StringIO
except ImportError:
    import StringIO

__all__ = ['GenericDumper', 'Dumper',
    'emit', 'dump', 'emit_documents', 'dump_documents']

class GenericDumper(_syck.Emitter):
    """
    GenericDumper dumps native Python objects into YAML documents.
    """

    def dump(self, object):
        """Dumps the given Python object as a YAML document."""
        self.emit(self._convert(object, {}))

    def _convert(self, object, object_to_node):
        if id(object) in object_to_node and self.allow_aliases(object):
            return object_to_node[id(object)][1]
        node = self.represent(object)
        object_to_node[id(object)] = object, node
        if node.kind == 'seq':
            for index in range(len(node.value)):
                item = node.value[index]
                node.value[index] = self._convert(item, object_to_node)
        elif node.kind == 'map':
            if isinstance(node.value, dict):
                for key in node.value.keys():
                    value = node.value[key]
                    del node.value[key]
                    node.value[self._convert(key, object_to_node)] =    \
                            self._convert(value, object_to_node)
            elif isinstance(node.value, list):
                for index in range(len(node.value)):
                    key, value = node.value[index]
                    node.value[index] = (self._convert(key, object_to_node),
                            self._convert(value, object_to_node))
#        # Workaround against a Syck bug:
#        if node.kind == 'scalar' and node.style not in ['1quote', '2quote'] \
#                and node.value and node.value[-1] in [' ', '\t']:
#            node.style = '2quote'
        return node

    def represent(self, object):
        """Represents the given Python object as a 'Node'."""
        if isinstance(object, dict):
            return _syck.Map(object.copy(), tag="tag:yaml.org,2002:map")
        elif isinstance(object, list):
            return _syck.Seq(object[:], tag="tag:yaml.org,2002:seq")
        else:
            return _syck.Scalar(str(object), tag="tag:yaml.org,2002:str")

    def allow_aliases(self, object):
        """Checks whether the given object can be aliased."""
        return True

class Dumper(GenericDumper):
    """
    Dumper dumps native Python objects into YAML documents.
    """

    INF = 1e300000
    inf_value = repr(INF)
    neginf_value = repr(-INF)
    nan_value = repr(INF/INF)

    def find_representer(self, object):
        """
        For the given object, find a method that can represent it as a 'Node'
        object.

        If the type of the object has the form 'package.module.type',
        find_representer() returns the method 'represent_package_module_type'.
        If this method does not exist, it checks the base types.
        """
        for object_type in type(object).__mro__:
            if object_type.__module__ == '__builtin__':
                name = object_type.__name__
            else:
                name = '%s.%s' % (object_type.__module__, object_type.__name__)
            method = 'represent_' + name.replace('.', '_')
            if hasattr(self, method):
                return getattr(self, method)

    def represent(self, object):
        """Represents the given Python object as a 'Node'."""
        representer = self.find_representer(object)
        if representer:
            return representer(object)
        else:
            return super(Dumper, self).represent(object)

    def represent_object(self, object):
        return _syck.Scalar(repr(object), tag="tag:yaml.org,2002:str")

    def represent_NoneType(self, object):
        return _syck.Scalar('~', tag="tag:yaml.org,2002:null")

    def represent_bool(self, object):
        return _syck.Scalar(repr(object), tag="tag:yaml.org,2002:bool")

    def represent_str(self, object):
        try:
            return _syck.Scalar(object.encode('ascii'), tag="tag:yaml.org,2002:str")
        except UnicodeDecodeError:
            try:
                return _syck.Scalar(unicode(object, 'utf-8').encode('utf-8'),
                        tag="tag:python.yaml.org,2002:str")
            except UnicodeDecodeError:
                return _syck.Scalar(object.encode('base64'),
                        tag="tag:yaml.org,2002:binary")

    def represent_unicode(self, object):
        try:
            return _syck.Scalar(object.encode('ascii'), tag="tag:python.yaml.org,2002:unicode")
        except UnicodeEncodeError:
            return _syck.Scalar(object.encode('utf-8'), tag="tag:yaml.org,2002:str")

    def represent_list(self, object):
        return _syck.Seq(object[:], tag="tag:yaml.org,2002:seq")

    def represent_dict(self, object):
        return _syck.Map(object.copy(), tag="tag:yaml.org,2002:map")

    def represent_int(self, object):
        return _syck.Scalar(repr(object), tag="tag:yaml.org,2002:int")

    def represent_float(self, object):
        value = repr(object)
        if value == self.inf_value:
            value = '.inf'
        elif value == self.neginf_value:
            value = '-.inf'
        elif value == self.nan_value:
            value = '.nan'
        return _syck.Scalar(value, tag="tag:yaml.org,2002:float")

    def represent_complex(self, object):
        if object.real != 0.0:
            value = '%s+%sj' % (repr(object.real), repr(object.imag))
        else:
            value = '%sj' % repr(object.imag)
        return _syck.Scalar(value, tag="tag:python.yaml.org,2002:complex")

    def represent_sets_Set(self, object):
        return _syck.Seq(list(object), tag="tag:yaml.org,2002:set")
    represent_set = represent_sets_Set

    def represent_datetime_datetime(self, object):
        return _syck.Scalar(object.isoformat(), tag="tag:yaml.org,2002:timestamp")

    def represent_long(self, object):
        return _syck.Scalar(repr(object), tag="tag:python.yaml.org,2002:long")

    def represent_tuple(self, object):
        return _syck.Seq(list(object), tag="tag:python.yaml.org,2002:tuple")

    def represent_type(self, object):
        name = '%s.%s' % (object.__module__, object.__name__)
        return _syck.Scalar('', tag="tag:python.yaml.org,2002:name:"+name)
    represent_classobj = represent_type
    represent_class = represent_type
    # TODO: Python 2.2 does not provide the module name of a function
    represent_function = represent_type
    represent_builtin_function_or_method = represent_type

    def represent_module(self, object):
        return _syck.Scalar('', tag="tag:python.yaml.org,2002:module:"+object.__name__)

    def represent_instance(self, object):
        cls = object.__class__
        class_name = '%s.%s' % (cls.__module__, cls.__name__)
        args = ()
        state = {}
        if hasattr(object, '__getinitargs__'):
            args = object.__getinitargs__()
        if hasattr(object, '__getstate__'):
            state = object.__getstate__()
        elif not hasattr(object, '__getinitargs__'):
            state = object.__dict__.copy()
        if not args and isinstance(state, dict):
            return _syck.Map(state.copy(),
                    tag="tag:python.yaml.org,2002:object:"+class_name)
        value = {}
        if args:
            value['args'] = list(args)
        if state or not isinstance(state, dict):
            value['state'] = state
        return _syck.Map(value,
                tag="tag:python.yaml.org,2002:new:"+class_name)

    def represent_object(self, object): # Do you understand this? I don't.
        cls = type(object)
        class_name = '%s.%s' % (cls.__module__, cls.__name__)
        args = ()
        state = {}
        if cls.__reduce__ is type.__reduce__:
            if hasattr(object, '__reduce_ex__'):
                reduce = object.__reduce_ex__(2)
                args = reduce[1][1:]
            else:
                reduce = object.__reduce__()
            if len(reduce) > 2:
                state = reduce[2]
            if state is None:
                state = {}
            if not args and isinstance(state, dict):
                return _syck.Map(state.copy(),
                        tag="tag:python.yaml.org,2002:object:"+class_name)
            if not state and isinstance(state, dict):
                return _syck.Seq(list(args),
                        tag="tag:python.yaml.org,2002:new:"+class_name)
            value = {}
            if args:
                value['args'] = list(args)
            if state or not isinstance(state, dict):
                value['state'] = state
            return _syck.Map(value,
                    tag="tag:python.yaml.org,2002:new:"+class_name)
        else:
            reduce = object.__reduce__()
            cls = reduce[0]
            class_name = '%s.%s' % (cls.__module__, cls.__name__)
            args = reduce[1]
            state = None
            if len(reduce) > 2:
                state = reduce[2]
            if state is None:
                state = {}
            if not state and isinstance(state, dict):
                return _syck.Seq(list(args),
                        tag="tag:python.yaml.org,2002:apply:"+class_name)
            value = {}
            if args:
                value['args'] = list(args)
            if state or not isinstance(state, dict):
                value['state'] = state
            return _syck.Map(value,
                    tag="tag:python.yaml.org,2002:apply:"+class_name)

    def represent__syck_Node(self, object):
        object_type = type(object)
        type_name = '%s.%s' % (object_type.__module__, object_type.__name__)
        state = []
        if hasattr(object_type, '__slotnames__'):
            for name in object_type.__slotnames__:
                value = getattr(object, name)
                if value:
                    state.append((name, value))
        return _syck.Map(state,
                tag="tag:python.yaml.org,2002:object:"+type_name)

    def allow_aliases(self, object):
        """Checks whether the given object can be aliased."""
        if object is None or type(object) in [int, bool, float]:
            return False
        if type(object) is str and (not object or object.isalnum()):
            return False
        if type(object) is tuple and not object:
            return False
        return True

def emit(node, output=None, Dumper=Dumper, **parameters):
    """
    Emits the given node to the output.

    If output is None, returns the produced YAML document.
    """
    if output is None:
        dumper = Dumper(StringIO.StringIO(), **parameters)
    else:
        dumper = Dumper(output, **parameters)
    dumper.emit(node)
    if output is None:
        return dumper.output.getvalue()

def dump(object, output=None, Dumper=Dumper, **parameters):
    """
    Dumps the given object to the output.

    If output is None, returns the produced YAML document.
    """
    if output is None:
        dumper = Dumper(StringIO.StringIO(), **parameters)
    else:
        dumper = Dumper(output, **parameters)
    dumper.dump(object)
    if output is None:
        return dumper.output.getvalue()

def emit_documents(nodes, output=None, Dumper=Dumper, **parameters):
    """
    Emits the list of nodes to the output.
    
    If output is None, returns the produced YAML document.
    """
    if output is None:
        dumper = Dumper(StringIO.StringIO(), **parameters)
    else:
        dumper = Dumper(output, **parameters)
    for node in nodes:
        dumper.emit(node)
    if output is None:
        return dumper.output.getvalue()

def dump_documents(objects, output=None, Dumper=Dumper, **parameters):
    """
    Dumps the list of objects to the output.
    
    If output is None, returns the produced YAML document.
    """
    if output is None:
        dumper = Dumper(StringIO.StringIO(), **parameters)
    else:
        dumper = Dumper(output, **parameters)
    for object in objects:
        dumper.dump(object)
    if output is None:
        return dumper.output.getvalue()


