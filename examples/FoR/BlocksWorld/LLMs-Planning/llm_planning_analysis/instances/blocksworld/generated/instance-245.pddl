(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects g l b j)
(:init 
(handempty)
(ontable g)
(ontable l)
(ontable b)
(ontable j)
(clear g)
(clear l)
(clear b)
(clear j)
)
(:goal
(and
(on g l)
(on l b)
(on b j)
)))