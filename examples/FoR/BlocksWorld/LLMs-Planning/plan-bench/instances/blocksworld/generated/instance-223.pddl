(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects d l i)
(:init 
(handempty)
(ontable d)
(ontable l)
(ontable i)
(clear d)
(clear l)
(clear i)
)
(:goal
(and
(on d l)
(on l i)
)))