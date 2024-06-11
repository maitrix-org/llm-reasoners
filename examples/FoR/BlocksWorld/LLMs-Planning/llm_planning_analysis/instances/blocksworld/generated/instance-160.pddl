(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects j g a)
(:init 
(handempty)
(ontable j)
(ontable g)
(ontable a)
(clear j)
(clear g)
(clear a)
)
(:goal
(and
(on j g)
(on g a)
)))