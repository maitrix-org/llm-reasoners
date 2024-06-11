(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects g b f)
(:init 
(handempty)
(ontable g)
(ontable b)
(ontable f)
(clear g)
(clear b)
(clear f)
)
(:goal
(and
(on g b)
(on b f)
)))