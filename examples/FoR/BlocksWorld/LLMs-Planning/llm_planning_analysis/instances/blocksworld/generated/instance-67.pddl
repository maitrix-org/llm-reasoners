(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects g l b)
(:init 
(handempty)
(ontable g)
(ontable l)
(ontable b)
(clear g)
(clear l)
(clear b)
)
(:goal
(and
(on g l)
(on l b)
)))