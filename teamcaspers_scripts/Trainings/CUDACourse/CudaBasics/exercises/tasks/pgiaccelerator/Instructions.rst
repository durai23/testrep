PGI Accelerator
===============

This is a simple implementation of a matrix-matrix multiplication in Fortran. 

A for loop in Fortran is called a do loop and it has the form

.. code-block:: Fortran

   do i=1, 5
   ...
   end do

The variable i takes the values 1, 2, 3, 4, 5. The first element of an array 
`A` is usually `A(1)`.

Todo
-----

#. Run ``make cpu`` to compile the CPU version of the code

#. Open the file mm.f90 and insert the PGI Accelerator directives. 

#. Run ``make acc`` to compile your changed code.

#. Run the programs. How fast are the codes? How many GFLOP/s do you get? Are those good values?

If you are having problems make sure that you are running the programs on a node with GPUs.

Extra credit
------------
Rewrite the program in C.
