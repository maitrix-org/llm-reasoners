(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects l g b)
(:init 
(handempty)
(ontable l)
(ontable g)
(ontable b)
(clear l)
(clear g)
(clear b)
)
(:goal
(and
(on l g)
(on g b)
)))