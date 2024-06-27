(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects d l j)
(:init 
(handempty)
(ontable d)
(ontable l)
(ontable j)
(clear d)
(clear l)
(clear j)
)
(:goal
(and
(on d l)
(on l j)
)))