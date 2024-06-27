(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects l h a b)
(:init 
(handempty)
(ontable l)
(ontable h)
(ontable a)
(ontable b)
(clear l)
(clear h)
(clear a)
(clear b)
)
(:goal
(and
(on l h)
(on h a)
(on a b)
)))