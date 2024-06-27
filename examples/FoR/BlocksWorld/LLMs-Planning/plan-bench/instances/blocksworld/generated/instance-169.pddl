(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects g b k)
(:init 
(handempty)
(ontable g)
(ontable b)
(ontable k)
(clear g)
(clear b)
(clear k)
)
(:goal
(and
(on g b)
(on b k)
)))