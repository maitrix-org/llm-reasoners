(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects j a g)
(:init 
(handempty)
(ontable j)
(ontable a)
(ontable g)
(clear j)
(clear a)
(clear g)
)
(:goal
(and
(on j a)
(on a g)
)))