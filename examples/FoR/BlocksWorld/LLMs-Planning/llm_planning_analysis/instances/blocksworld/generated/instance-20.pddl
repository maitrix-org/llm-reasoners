(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects d b f)
(:init 
(handempty)
(ontable d)
(ontable b)
(ontable f)
(clear d)
(clear b)
(clear f)
)
(:goal
(and
(on d b)
(on b f)
)))