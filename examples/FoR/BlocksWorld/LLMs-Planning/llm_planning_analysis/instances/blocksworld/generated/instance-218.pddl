(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects d l a)
(:init 
(handempty)
(ontable d)
(ontable l)
(ontable a)
(clear d)
(clear l)
(clear a)
)
(:goal
(and
(on d l)
(on l a)
)))