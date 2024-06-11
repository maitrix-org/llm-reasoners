(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects k j b)
(:init 
(handempty)
(ontable k)
(ontable j)
(ontable b)
(clear k)
(clear j)
(clear b)
)
(:goal
(and
(on k j)
(on j b)
)))