(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects d h a l)
(:init 
(handempty)
(ontable d)
(ontable h)
(ontable a)
(ontable l)
(clear d)
(clear h)
(clear a)
(clear l)
)
(:goal
(and
(on d h)
(on h a)
(on a l)
)))