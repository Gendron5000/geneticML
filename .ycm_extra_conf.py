def Settings( **kwargs ):
  return {
    'flags': [ '-x', 'c++', '-std=c++17', '-lmlpack', '-lblas', '-larmadillo' ],
  }
