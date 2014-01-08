"""
YAML is a data serialization format designed for human readability and
interaction with scripting languages.

Syck is an extension for reading and writing YAML in scripting languages.

PySyck provides Python bindings for Syck YAML parser and emitter.

To start working with PySyck, import the package 'syck':
>>> from syck import *

To parse a YAML document into a Python object, use the function 'load()':
>>> load('''
... - Mark McGwire
... - Sammy Sosa
... - Ken Griffey
... ''')
['Mark McGwire', 'Sammy Sosa', 'Ken Griffey']

To emit a Python object into a YAML document, use the function 'dump()':
>>> print dump(['Mark McGwire', 'Sammy Sosa', 'Ken Griffey'])
---
- Mark McGwire
- Sammy Sosa
- Ken Griffey

You may get access to the YAML parser tree using the function 'parse()':
>>> root_node = parse('''
... - Mark McGwire
... - Sammy Sosa
... - Ken Griffey
... ''')
>>> root_node
<_syck.Seq object at 0xb7a1f874>
>>> root_node.kind
'seq'
>>> root_node.value
[<_syck.Scalar object at 0xb7a1e5fc>, <_syck.Scalar object at 0xb7a1e65c>, <_syck.Scalar object at 0xb7a1e6bc>]

You may now use the function 'emit()' to obtain the YAML document again:
>>> print emit(root_node)
---
- Mark McGwire
- Sammy Sosa
- Ken Griffey

What do you get if you apply the function 'dump()' to root_node? Let's try it:
>>> print dump(root_node)
--- !python/object:_syck.Seq
value:
- !python/object:_syck.Scalar
  value: Mark McGwire
  tag: tag:yaml.org,2002:str
- !python/object:_syck.Scalar
  value: Sammy Sosa
  tag: tag:yaml.org,2002:str
- !python/object:_syck.Scalar
  value: Ken Griffey
  tag: tag:yaml.org,2002:str

As you can see, PySyck allow you to represent complex Python objects.

You can also dump the generated YAML output into any file-like object:
>>> import os
>>> stream = os.tmpfile()
>>> object = ['foo', 'bar', ['baz']]
>>> dump(object, stream)
>>> stream.seek(0)
>>> print stream.read()
---
- foo
- bar
- - baz

To load several documents from a single YAML stream, use the function
'load_documents()':
>>> source = '''
... ---
... american:
...   - Boston Red Sox
...   - Detroit Tigers
...   - New York Yankees
... national:
...   - New York Mets
...   - Chicago Cubs
...   - Atlanta Braves
... ---
... - [name        , hr, avg  ]
... - [Mark McGwire, 65, 0.278]
... - [Sammy Sosa  , 63, 0.288]
... '''
>>> for document in load_documents(source):
...     print document
...
{'national': ['New York Mets', 'Chicago Cubs', 'Atlanta Braves'], 'american': ['Boston Red Sox', 'Detroit Tigers', 'New York Yankees']}
[['name', 'hr', 'avg'], ['Mark McGwire', 65, 0.27800000000000002], ['Sammy Sosa', 63, 0.28799999999999998]]

See the source code for more details.
"""


from _syck import *
from loaders import *
from dumpers import *

