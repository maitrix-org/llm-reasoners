(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects g d l i)
(:init 
(handempty)
(ontable g)
(ontable d)
(ontable l)
(ontable i)
(clear g)
(clear d)
(clear l)
(clear i)
)
(:goal
(and
(on g d)
(on d l)
(on l i)
)))